import os 
from dotenv import load_dotenv

load_dotenv()

# POSTGRES_USER = os.getenv("POSTGRES_USER")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
# POSTGRES_DB = os.getenv("POSTGRES_DB")
# POSTGRES_HOST = "localhost"
# POSTGRES_PORT = "5432"

# DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:supersecretpassword@localhost:5432/rag_db"
)