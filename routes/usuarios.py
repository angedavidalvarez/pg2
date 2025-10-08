from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_connection

router = APIRouter()


class Usuario(BaseModel):
    id: int
    correo: str
    rol: str

class UsuarioCompleto(BaseModel):
    id: int
    correo: str
    rol: str
    nombre: str

class UsuarioCrear(BaseModel):
    nombre: str
    correo: str
    contrasena: str
    rol: int  # 1: Admin, 2: Usuario (o como lo tengas)


# Endpoint para obtener todos los usuarios con todos los campos
@router.get("/usuario/completo", response_model=list[UsuarioCompleto])
def obtener_usuarios_completos():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, correo, rol, nombre FROM usuarios")
    resultados = cursor.fetchall()
    cursor.close()
    db.close()

    usuarios = []
    for row in resultados:
        usuarios.append({
            "id": row["id"],
            "correo": row["correo"],
            "rol": "Administrador" if str(row["rol"]) == "1" else "Operador",
            "nombre": row["nombre"]
        })
    return usuarios

# Endpoint para crear un usuario
@router.post("/usuario")
def crear_usuario(usuario: dict):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO usuarios (correo, contrasena, rol, nombre) VALUES (%s, %s, %s, %s)",
            (usuario["correo"], usuario["contrasena"], 1 if usuario["rol"] == "Administrador" else 2, usuario.get("nombre", ""))
        )
        db.commit()
        return {"mensaje": "Usuario creado correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()

# Endpoint para editar un usuario (solo rol)
@router.put("/usuario/{usuario_id}")
def editar_usuario(usuario_id: int, datos: dict):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE usuarios SET rol = %s WHERE id = %s",
            (1 if datos["rol"] == "Administrador" else 2, usuario_id)
        )
        db.commit()
        return {"mensaje": "Usuario actualizado correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()

# Endpoint para eliminar un usuario
@router.delete("/usuario/{usuario_id}")
def eliminar_usuario(usuario_id: int):
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM usuarios WHERE id = %s", (usuario_id,))
        db.commit()
        return {"mensaje": "Usuario eliminado correctamente"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        db.close()


@router.get("/usuario", response_model=list[Usuario])
def obtener_usuarios():
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, correo, rol FROM usuarios")
    resultados = cursor.fetchall()
    cursor.close()
    db.close()

    usuarios = []
    for row in resultados:
        usuarios.append({
            "id": row["id"],
            "correo": row["correo"],
            "rol": row["rol"]
        })

    return usuarios