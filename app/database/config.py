import os
from dotenv import load_dotenv

load_dotenv()

# Chain of fallbacks: Docker Network -> Local .env -> Hardcoded Localhost
DATABASE_URL = (
    os.environ.get("DOCKER_DATABASE_URL") or 
    os.environ.get("DATABASE_URL") or 
    "postgresql+psycopg://postgres:supersecretpassword@localhost:5432/rag_db"
)