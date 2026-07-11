from fastapi import FastAPI

app = FastAPI(title="Financial RAG Engine API", version="1.0")

@app.get("/")
async def root():
    return {"message": "Financial RAG Engine is running."}