from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from database import get_connection
import io
from reportlab.pdfgen import canvas

router = APIRouter()

class Incidente(BaseModel):
    id: int
    id_evento: int
    id_usuario: int
    estado: str
    fecha_envio: str

@router.get("/incidentes", response_model=List[Incidente])
def obtener_incidentes():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT a.id, a.id_evento, a.id_usuario, a.estado, a.fecha_envio
            FROM alertas a
            JOIN eventos e ON a.id_evento = e.id
            ORDER BY a.fecha_envio DESC
        """)
        resultados = cursor.fetchall()
        return resultados
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        db.close()


@router.get("/incidentes/pdf")
def descargar_incidentes_pdf():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT a.id, a.id_evento, a.id_usuario, a.estado, a.fecha_envio
            FROM alertas a
            JOIN eventos e ON a.id_evento = e.id
            ORDER BY a.fecha_envio DESC
        """)
        incidentes = cursor.fetchall()

        # Crear PDF en memoria
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        y = 800
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, "Historial de Incidentes")
        y -= 30

        for inc in incidentes:
            fecha = inc["fecha_envio"].strftime("%Y-%m-%d %H:%M:%S") if inc["fecha_envio"] else ""
            p.setFont("Helvetica", 10)
            p.drawString(50, y, f"ID: {inc['id']} | ID Evento: {inc['id_evento']} | ID Usuario: {inc['id_usuario']} | Estado: {inc['estado']} | Fecha: {fecha}")
            y -= 20
            if y < 50:  # nueva página
                p.showPage()
                y = 800

        p.save()
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=incidentes.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        db.close()
