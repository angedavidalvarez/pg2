# routes/camaras.py - Optimizado producción 40+ cámaras
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from database import get_connection
from torchvision import models, transforms
import logging
from collections import deque
import numpy as np
import time

# ===================== ROUTER =====================
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camaras")

# ===================== GPU =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== MODELO =====================
class FightDetectionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super(FightDetectionModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        with torch.no_grad():
            features = self.cnn(x).squeeze(-1).squeeze(-1)
        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ===================== CARGAR MODELO =====================
try:
    # Para máxima velocidad, si tienes TorchScript: model = torch.jit.load("fight_detection_model_scripted.pt").to(device)
    model = FightDetectionModel().to(device)
    state_dict = torch.load("fight_detection_model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")

# ===================== GLOBAL =====================
camara_frames = {}     # último frame con overlay
camara_locks = {}      # lock por cámara
camara_buffers = {}    # buffer circular deque por cámara
sequence_length = 16
skip_frames = 2

# ===================== TRANSFORM =====================
preprocess_transform = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def preprocess(frame):
    frame = cv2.resize(frame, (224,224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.tensor(frame/255., dtype=torch.float32).permute(2,0,1)
    return preprocess_transform(frame_tensor)

# ===================== HILOS DE CAPTURA =====================
def procesar_camara(camara_id: int, url: str, resolution=(640,480)):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if not cap.isOpened():
        logger.error(f"No se pudo abrir la cámara {camara_id}")
        return

    frame_count = 0
    buffer = deque(maxlen=sequence_length)
    camara_buffers[camara_id] = buffer
    camara_locks[camara_id] = threading.Lock()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Stream cerrado para cámara {camara_id}")
            cap.release()
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        buffer.append(preprocess(frame))
        # Guardar último frame original para MJPEG
        with camara_locks[camara_id]:
            camara_frames[camara_id] = cv2.imencode('.jpg', frame)[1].tobytes()

def iniciar_hilo_camara(camara_id: int, url: str):
    t = threading.Thread(target=procesar_camara, args=(camara_id,url), daemon=True)
    t.start()
    logger.info(f"Hilo de captura iniciado para cámara {camara_id}")

# ===================== HILO DE INFERENCIA BATCH =====================
def hilo_inferencia_batch():
    while True:
        batch = []
        camaras_ready = []
        for cam_id, buffer in camara_buffers.items():
            with camara_locks[cam_id]:
                if len(buffer) == sequence_length:
                    batch.append(torch.stack(list(buffer)))
                    camaras_ready.append(cam_id)

        if batch:
            frames_batch = torch.stack(batch).to(device)
            with torch.no_grad():
                outputs = model(frames_batch)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

            # actualizar overlay en frames
            for cam_id, pred_class in zip(camaras_ready, preds):
                label = "Pelea detectada" if pred_class else "Sin pelea"
                with camara_locks[cam_id]:
                    frame_bytes = camara_frames[cam_id]
                    frame_array = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                    color = (0,0,255) if pred_class else (0,255,0)
                    cv2.putText(frame_array, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    camara_frames[cam_id] = cv2.imencode('.jpg', frame_array)[1].tobytes()
        time.sleep(0.01)  # evita 100% CPU

# iniciar hilo batch
inference_thread = threading.Thread(target=hilo_inferencia_batch, daemon=True)
inference_thread.start()

# ===================== MODELOS Pydantic =====================
class Camara(BaseModel):
    id: int
    nombre: str
    ubicacion: str
    url_stream: str
    estado: str

class CamaraCrear(BaseModel):
    nombre: str
    ubicacion: str
    url_stream: str
    estado: str = "activa"

# ===================== ENDPOINTS =====================
@router.get("/camaras/{camara_id}/analizar_mjpeg")
def analizar_mjpeg(camara_id: int):
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT url_stream, estado FROM camaras WHERE id=%s", (camara_id,))
    camara = cursor.fetchone()
    cursor.close()
    db.close()

    if not camara:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    if camara.get("estado") == "inactiva":
        raise HTTPException(status_code=403, detail="Cámara desactivada")

    url = camara["url_stream"]
    if camara_id not in camara_frames:
        iniciar_hilo_camara(camara_id, url)

    def mjpeg_generator():
        while True:
            if camara_id in camara_frames:
                with camara_locks[camara_id]:
                    frame = camara_frames[camara_id]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

# ===================== CRUD =====================
@router.get("/camaras", response_model=list[Camara])
def listar_camaras():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM camaras")
    camaras = cursor.fetchall()
    cursor.close()
    db.close()
    return camaras

@router.post("/camaras")
def agregar_camara(camara: CamaraCrear):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO camaras (nombre, ubicacion, url_stream, estado) VALUES (%s, %s, %s, %s)",
            (camara.nombre, camara.ubicacion, camara.url_stream, camara.estado)
        )
        db.commit()
        return {"mensaje": "Cámara agregada correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()

@router.put("/camaras/{camara_id}")
def editar_camara(camara_id: int, camara: CamaraCrear):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE camaras SET nombre=%s, ubicacion=%s, url_stream=%s, estado=%s WHERE id=%s",
            (camara.nombre, camara.ubicacion, camara.url_stream, camara.estado, camara_id)
        )
        db.commit()
        return {"mensaje": "Cámara actualizada correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()

@router.delete("/camaras/{camara_id}")
def eliminar_camara(camara_id: int):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM camaras WHERE id=%s", (camara_id,))
        db.commit()
        return {"mensaje": "Cámara eliminada correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()

# ===================== PROBAR CONEXIÓN =====================
@router.get("/camaras/probar")
def probar_camara(url: str):
    logger.info(f"Intentando conectar a la cámara: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return JSONResponse(content={"conectado": False, "mensaje": "No se pudo conectar"}, status_code=200)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse(content={"conectado": False, "mensaje": "No se pudo obtener imagen"}, status_code=200)
    return JSONResponse(content={"conectado": True, "mensaje": "Conexión exitosa"}, status_code=200)
