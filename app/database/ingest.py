import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

#Import the necessary modules for database connection and configuration
from app.database.config import DATABASE_URL

# Load environment variables (like OpenAI key)
load_dotenv()

def ingest_documents():
    print("1.Starting document ingestion process...")
    # Example document loading (replace with actual logic)
    loader = TextLoader("data/apple_q3.txt")
    documents = loader.load()

    print(f"2.Loaded {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    print(f"-> Split into {len(chunks)} chunks.")

    print("3.Creating embeddings for the document chunks...")
    #Intialize the OpenAI embeddings model
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # Send chunks to the embedding model and store the embeddings in the PostgreSQL 
    PGVector.from_documents(
        documents=chunks,
        embedding=embedding,
        connection_string=DATABASE_URL,
        collection_name="financial_reports"
    )

    print("✅ Document ingestion and embedding process completed successfully. Data is now stored in the PostgreSQL database.")
    
if __name__ == "__main__":
    ingest_documents()