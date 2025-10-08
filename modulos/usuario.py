# modulos/usuarios.py
from fastapi import APIRouter
from pydantic import BaseModel
from database import get_connection

router = APIRouter()

class Usuario(BaseModel):
    id: int
    nombre: str
    correo: str
    rol: str

@router.get("/usuarios", response_model=list[Usuario])
def obtener_usuarios():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, nombre, correo, rol FROM usuarios")
    resultados = cursor.fetchall()
    cursor.close()
    db.close()

    usuarios = []
    for row in resultados:
        usuarios.append({
            "id": row["id"],
            "nombre": row["nombre"],
            "correo": row["correo"],
            "rol": "Admin" if str(row["rol"]) == "1" else "Usuario"
        })

    return usuarios




