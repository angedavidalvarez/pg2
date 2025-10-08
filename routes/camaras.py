
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import cv2
import numpy as np
from fastapi.responses import StreamingResponse, JSONResponse
from database import get_connection
import requests

router = APIRouter()

# Endpoint proxy MJPEG para cámaras tipo IP Webcam
@router.get("/camaras/{camara_id}/mjpeg")
def proxy_mjpeg(camara_id: int, request: Request):
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT url_stream, estado FROM camaras WHERE id=%s", (camara_id,))
    camara = cursor.fetchone()
    cursor.close()
    db.close()
    if not camara:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    if camara.get("estado") == "inactiva":
        raise HTTPException(status_code=403, detail="La cámara está desactivada")
    url = camara["url_stream"]
    # Asegurarse de que termina en /video
    if url.endswith('/'):
        url_video = url + 'video'
    else:
        url_video = url + '/video'
    # Validar protocolo soportado
    if not (url_video.startswith('http://') or url_video.startswith('https://')):
        raise HTTPException(status_code=400, detail="Solo se soportan cámaras con URL http:// o https:// para transmisión MJPEG. Protocolo no soportado: " + url_video)
    verify_ssl = not url_video.startswith('https://') or False
    try:
        r = requests.get(url_video, stream=True, verify=verify_ssl, timeout=5)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="La cámara no respondió a tiempo (timeout)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al conectar con la cámara: {str(e)}")
    content_type = r.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=--myboundary')
    def iter_mjpeg():
        try:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        finally:
            r.close()
    return StreamingResponse(iter_mjpeg(), media_type=content_type)

# Endpoint para consultar el estado de todas las cámaras
@router.get("/camaras/estado")
def estado_camaras():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, nombre, url_stream FROM camaras")
    camaras = cursor.fetchall()
    cursor.close()
    db.close()
    resultado = []
    for cam in camaras:
        cap = cv2.VideoCapture(cam["url_stream"])
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            estado = "conectada" if ret else "desconectada"
        else:
            estado = "desconectada"
        resultado.append({
            "id": cam["id"],
            "nombre": cam["nombre"],
            "estado": estado
        })
    return resultado

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

# Listar cámaras
@router.get("/camaras", response_model=list[Camara])
def listar_camaras():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM camaras")
    camaras = cursor.fetchall()
    cursor.close()
    db.close()
    return camaras


# Agregar cámara
@router.post("/camaras")
def agregar_camara(camara: dict):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO camaras (nombre, ubicacion, url_stream, estado) VALUES (%s, %s, %s, %s)",
            (camara["nombre"], camara["ubicacion"], camara["url_stream"], camara.get("estado", "activa"))
        )
        db.commit()
        return {"mensaje": "Cámara agregada correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()


# Editar cámara
@router.put("/camaras/{camara_id}")
def editar_camara(camara_id: int, camara: dict):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE camaras SET nombre=%s, ubicacion=%s, url_stream=%s, estado=%s WHERE id=%s",
            (camara["nombre"], camara["ubicacion"], camara["url_stream"], camara.get("estado", "activa"), camara_id)
        )
        db.commit()
        return {"mensaje": "Cámara actualizada correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()


# Eliminar cámara
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

# Probar conexión a cámara y obtener un frame
@router.get("/camaras/{camara_id}/frame")
def obtener_frame_camara(camara_id: int):
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT url_stream FROM camaras WHERE id=%s", (camara_id,))
    camara = cursor.fetchone()
    cursor.close()
    db.close()
    if not camara:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    url = camara["url_stream"]
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        # Intentar obtener una imagen fija si es HTTP (muchas cámaras IP tipo Android solo dan /shot.jpg)
        import requests
        try:
            if url.endswith('/'):
                url_img = url + 'shot.jpg'
            else:
                url_img = url + '/shot.jpg'
            resp = requests.get(url_img, timeout=3)
            if resp.status_code == 200:
                return StreamingResponse(iter([resp.content]), media_type="image/jpeg")
            else:
                raise HTTPException(status_code=400, detail=f"No se pudo conectar a la cámara ni obtener imagen fija: {resp.status_code}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo conectar a la cámara ni obtener imagen fija: {str(e)}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=400, detail="No se pudo obtener imagen de la cámara (stream abierto pero sin frame)")
    _, img_encoded = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([img_encoded.tobytes()]),
        media_type="image/jpeg"
    )

# Probar conexión a cámara (sin obtener frame)
import logging

@router.get("/camaras/probar")
def probar_camara(url: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("camaras")
    logger.info(f"Intentando conectar a la cámara: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir la conexión con la cámara: {url}")
        return JSONResponse(content={"conectado": False, "mensaje": "No se pudo conectar"}, status_code=200)
    ret, frame = cap.read()
    if not ret:
        logger.error(f"No se pudo obtener imagen de la cámara: {url}")
        cap.release()
        return JSONResponse(content={"conectado": False, "mensaje": "No se pudo obtener imagen"}, status_code=200)
    logger.info(f"Conexión exitosa y frame recibido de la cámara: {url}")
    cap.release()
    return JSONResponse(content={"conectado": True, "mensaje": "Conexión exitosa"}, status_code=200)
