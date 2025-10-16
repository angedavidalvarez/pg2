from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from database import get_connection
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

router = APIRouter()

@router.get("/reportes")
async def get_reportes():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, id_camara, tipo_evento, descripcion, fecha_hora FROM eventos ORDER BY fecha_hora DESC")
        reportes = cursor.fetchall()
        for r in reportes:
            r["fecha_hora"] = r["fecha_hora"].strftime("%Y-%m-%d %H:%M:%S") if r["fecha_hora"] else ""
        return reportes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/reportes/pdf")
def descargar_reportes_pdf():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, id_camara, tipo_evento, descripcion, fecha_hora FROM eventos ORDER BY fecha_hora DESC")
        reportes = cursor.fetchall()

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 20 * mm
        y_start = height - margin
        y = y_start
        line_height = 10

        # Encabezado
        def dibujar_encabezado():
            nonlocal y
            p.setFont("Helvetica-Bold", 14)
            p.drawString(margin, y, "Reportes de Eventos")
            y -= 20
            p.setFont("Helvetica-Bold", 10)
            p.drawString(margin, y, "ID")
            p.drawString(margin + 15*mm, y, "ID Cámara")
            p.drawString(margin + 35*mm, y, "Tipo Evento")
            p.drawString(margin + 65*mm, y, "Fecha y Hora")
            p.drawString(margin + 100*mm, y, "Descripción")
            y -= 12
            p.line(margin, y, width - margin, y)
            y -= 10

        dibujar_encabezado()
        p.setFont("Helvetica", 10)

        for r in reportes:
            if y < 30:  # nueva página
                p.showPage()
                y = y_start
                dibujar_encabezado()
                p.setFont("Helvetica", 10)

            # Manejo de descripción larga
            text_obj = p.beginText(margin + 100*mm, y)
            text_obj.setFont("Helvetica", 10)
            for line in r["descripcion"].split("\n"):
                text_obj.textLine(line)
            p.drawText(text_obj)

            p.drawString(margin, y, str(r["id"]))
            p.drawString(margin + 15*mm, y, str(r["id_camara"]))
            p.drawString(margin + 35*mm, y, r["tipo_evento"])
            fecha = r["fecha_hora"].strftime("%Y-%m-%d %H:%M:%S") if r["fecha_hora"] else ""
            p.drawString(margin + 65*mm, y, fecha)

            y -= max(line_height, 10 * len(r["descripcion"].split("\n")))

        p.save()
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=reportes_eventos.pdf"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
