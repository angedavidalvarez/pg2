# ...existing code...
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "videovigilancia"),
            connect_timeout=5
        )
    except mysql.connector.Error as e:
        raise RuntimeError(f"DB connection error: {e}") from e
# ...existing code...