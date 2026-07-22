import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from app.database.config import DATABASE_URL
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from app.database.ingest import fetch_and_embed_latest_10q

load_dotenv()

def run_daily_ingestion():
    print("Executing scheduled SEC ingestion...")
    try:
        fetch_and_embed_latest_10q("AAPL")
    except Exception as e:
        print(f"Scheduled ingestion failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background scheduler
    scheduler = BackgroundScheduler()
    # Schedule the job to run every day at 17:00 (5 PM)
    scheduler.add_job(run_daily_ingestion,'cron',hour=17,minute=0)
    scheduler.start()
    print("Background scheduler started for daily SEC ingestion at 5 PM.")

    yield # This is where the application runs

    #Shutdown: Stop the background scheduler
    scheduler.shutdown()
    print("Background scheduler stopped.")


app = FastAPI(
    title="Financial RAG Engine API",
    description="An AI-powered API that answers financial questions and provides source evidence.",
    version="1.1.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list[dict]

@app.get("/")
async def health_check():
    return {"status": "Financial RAG Engine is running."}

# --- RAG Pipeline Initialization ---
# We initialize this outside the endpoint so the server doesn't have to 
# reconnect to the database and re-download the model on every single request.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = PGVector(
    connection=DATABASE_URL,
    embeddings=embeddings,
    collection_name="financial_reports",
    use_jsonb=True,
)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are a helpful financial assistant. Answer the question using ONLY the following context.
If you don't know the answer based on the context, just say that you don't know.

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    formatted = []
    for doc in docs:
        # Grab the structural headers from metadata
        h1 = doc.metadata.get("Heading 1", "")
        h2 = doc.metadata.get("Heading 2", "")
        h3 = doc.metadata.get("Heading 3", "")

        # Combine them (e.g., "Finacial statements > Operations")
        headers = " > ".join(filter(None,[h1,h2,h3]))

        if headers:
            formatted.append(f"[Section: {headers}]\n{doc.page_content}")
        else:
            formatted.append(doc.page_content)

    return "\n\n".join(formatted)



    # return "\n\n".join(doc.page_content for doc in docs)

# This is the final, efficient RAG chain. It retrieves documents once and
# reuses them for both context formatting and for passing through as sources.
rag_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(source_documents=retriever)
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["source_documents"]))
    | {
        "answer": generation_chain,
        "source_documents": lambda x: x["source_documents"],
    }
)

@app.post("/query",response_model=QueryResponse)
async def ask_finacial_question(request: QueryRequest):
    try:
        result = rag_chain.invoke(request.question)
        # result['source_documents'] = [doc.page_content for doc in result['source_documents']]
        # Return both the content AND the metadata to the frontend
        result['source_documents'] = [
            {"content": doc.page_content, "metadata": doc.metadata} 
            for doc in result['source_documents']
        ]
        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))