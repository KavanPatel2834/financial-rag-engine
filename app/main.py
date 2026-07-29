import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, Table, MetaData

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_postgres import PGVector
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
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

# 1. Vector Store Retriever (Semantic Search)
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# 2. Keyword Retriever (BM25)
# We need to fetch all docs to initialize this. This is a one-time setup cost.
print("Initializing keyword retriever...")

# The PGVector class does not have a 'get_documents' method or a public '_table' attribute.
# We can fetch all documents by reflecting the table from the database using SQLAlchemy.
all_docs = []
metadata = MetaData()
with db._engine.connect() as conn:
    # Reflect the table structure from the database using the known collection name
    collection_table = Table(
        db.collection_name, 
        metadata, 
        autoload_with=conn
    )
    # Execute a select query on the reflected table
    result = conn.execute(select(collection_table))
    print("WARNING: 'financial_reports' table not found. BM25Retriever will be initialized with no documents.")
    print("Please run the ingestion script (e.g., 'python -m app.database.ingest') to populate the database.")

keyword_retriever = BM25Retriever.from_documents(all_docs)
keyword_retriever.k = 5

# 3. Manual Hybrid Search with Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    """
    Merges multiple lists of ranked documents using Reciprocal Rank Fusion,
    preserving the original Document objects.
    """
    fused_scores = {}
    doc_map = {}  # Maps page_content to the full Document object

    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            content = doc.page_content
            if content not in doc_map:
                doc_map[content] = doc  # Store the first-seen Document object

            if content not in fused_scores:
                fused_scores[content] = 0
            fused_scores[content] += 1 / (rank + k)

    reranked_results = [
        (doc_map[content], score) for content, score in fused_scores.items()
    ]
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_results] # Return only the sorted documents

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are a helpful financial assistant. Answer the question using ONLY the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    """Prepares the document context for the LLM."""
    formatted = []
    for doc in docs:
        # Grab the structural headers from metadata
        h1 = doc.metadata.get("Heading 1", "")
        h2 = doc.metadata.get("Heading 2", "")
        h3 = doc.metadata.get("Heading 3", "")
        headers = " > ".join(filter(None, [h1, h2, h3]))
        if headers:
            formatted.append(f"[Section: {headers}]\n{doc.page_content}")
        else:
            formatted.append(doc.page_content)
    return "\n\n".join(formatted)

# This runnable performs the hybrid search and returns the fused documents.
retriever_chain = (
    RunnableParallel(vector=vector_retriever, keyword=keyword_retriever)
    | (lambda x: reciprocal_rank_fusion([x["vector"], x["keyword"]]))
)

# This is the final, stateless RAG chain from the end of Task 5.
rag_chain = (
    {
        "source_documents": retriever_chain,
        "question": RunnablePassthrough()
    }
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
        # Convert Document objects to dictionaries for the JSON response
        source_docs_as_dicts = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in result["source_documents"]
        ]
        return QueryResponse(answer=result["answer"], source_documents=source_docs_as_dicts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))