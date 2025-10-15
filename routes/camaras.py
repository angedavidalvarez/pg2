import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from database import get_connection
from torchvision import transforms, models
import logging
import requests

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camaras")

# ===================== MODELO =====================
class FightDetectionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super(FightDetectionModel, self).__init__()
        # ResNet18 como extractor de features
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # sin fc final
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        with torch.no_grad():
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # [b*t, 512]
        features = features.view(b, t, -1)  # [b, t, 512]
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ===================== CARGAR MODELO =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FightDetectionModel()
try:
    state_dict = torch.load("fight_detection_model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)  # ignora keys faltantes
    model.to(device)
    model.eval()
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")

# ===================== TRANSFORM =====================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== GLOBAL =====================
camara_frames = {}  # Último frame procesado por cámara
camara_locks = {}   # Locks para acceso thread-safe

# ===================== FUNCIONES =====================
def procesar_camara(camara_id: int, url: str):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir la cámara {camara_id}: {url}")
        return

    sequence_length = 16
    buffer_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            logger.warning(f"Stream cerrado para cámara {camara_id}")
            break

        buffer_frames.append(frame)
        if len(buffer_frames) > sequence_length:
            buffer_frames.pop(0)

        if len(buffer_frames) == sequence_length:
            frames_tensor = torch.stack([transform(f) for f in buffer_frames])
            frames_tensor = frames_tensor.unsqueeze(0).to(device)  # [1, seq_len, C, H, W]
            with torch.no_grad():
                outputs = model(frames_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                label = "Pelea detectada" if pred_class == 1 else "Sin pelea"

            frame_overlay = buffer_frames[-1].copy()
            color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)
            cv2.putText(frame_overlay, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            with camara_locks[camara_id]:
                camara_frames[camara_id] = cv2.imencode('.jpg', frame_overlay)[1].tobytes()

def iniciar_hilo_camara(camara_id: int, url: str):
    if camara_id not in camara_locks:
        camara_locks[camara_id] = threading.Lock()
    t = threading.Thread(target=procesar_camara, args=(camara_id, url), daemon=True)
    t.start()
    logger.info(f"Hilo de procesamiento iniciado para cámara {camara_id}")

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
    
    return StreamingResponse(mjpeg_generator(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

# ===================== CRUD DE CÁMARAS =====================
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
