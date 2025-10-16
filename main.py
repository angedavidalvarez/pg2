from fastapi import FastAPI
from routes import auth, usuarios
from routes import camaras
from routes import reportes
from fastapi.middleware.cors import CORSMiddleware
from routes import incidentes
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router) 
app.include_router(usuarios.router)
app.include_router(camaras.router)
app.include_router(reportes.router)
app.include_router(incidentes.router) 

