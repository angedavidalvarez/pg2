from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_connection

router = APIRouter()

class LoginRequest(BaseModel):
    correo: str
    contrasena: str

@router.post("/login")
def login(data: LoginRequest):
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarios WHERE correo=%s AND contrasena=%s", (data.correo, data.contrasena))
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    return {"mensaje": "Login exitoso", "rol": result["rol"]}
