import os
from dotenv import load_dotenv

load_dotenv()

# Default to localhost for local development
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:supersecretpassword@localhost:5432/rag_db")

# If running inside a Docker container, use the internal network service name 'db'
if os.path.exists("/.dockerenv"):
    print("Running inside Docker! Using the internal network.")
    DATABASE_URL = "postgresql+psycopg://postgres:supersecretpassword@db:5432/rag_db"
else:
    print("Running locally! Using localhost.")