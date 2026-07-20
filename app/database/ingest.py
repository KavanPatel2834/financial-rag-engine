import os
from dotenv import load_dotenv

# SEC EDGAR Downloader
from edgar import Company, set_identity

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from app.database.config import DATABASE_URL
# load_dotenv()

# # The same robust Docker bypass we used in main.py
# if os.path.exists("/.dockerenv"):
#     print("Running inside Docker! Using the internal network.")
#     DATABASE_URL = "postgresql+psycopg://postgres:supersecretpassword@db:5432/rag_db"
# else:
#     print("Running locally! Using localhost.")
#     DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:supersecretpassword@localhost:5432/rag_db")


# def fetch_and_ingest_sec_filing(ticker: str, form_type: str = "10-Q"):
#     """
#     Connects to the SEC, downloads the latest filing for a given ticker,
#     and ingests it into our PGVector database.
#     """
#     print(f"Connecting to SEC EDGAR to fetch the latest {form_type} for {ticker}...")
    
#     # 1. Set SEC Identity (MANDATORY)
#     # The SEC blocks requests that don't declare a User-Agent. 
#     # Feel free to change this to your actual name and email.
#     set_identity("Kavan Patel kavanpatel0564@gmail.com")
    
#     # 2. Fetch the Company and the Filing
#     try:
#         company = Company(ticker)
#         # Get all filings of the requested form type, then grab the latest (1)
#         filings = company.get_filings(form=form_type)
#         latest_filing = filings[0] # The most recent one
        
#         print(f"Found filing: {latest_filing.form} filed on {latest_filing.filing_date}")
        
#         # Extract the raw text from the filing
#         raw_text = latest_filing.text()
#         print(f"Successfully downloaded {len(raw_text)} characters of text.")
        
#     except Exception as e:
#         print(f"Failed to fetch data from the SEC: {e}")
#         return

#     # 3. Create a LangChain Document and Chunk It
#     print("Chunking the financial document...")
    
#     # Attach metadata so the LLM knows exactly where this data came from
#     doc = Document(
#         page_content=raw_text,
#         metadata={
#             "source": f"SEC EDGAR",
#             "ticker": ticker,
#             "form": latest_filing.form,
#             "filing_date": latest_filing.filing_date,
#             "accession_number": latest_filing.accession_no
#         }
#     )
    
#     # Split the massive document into smaller pieces
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = text_splitter.split_documents([doc])
#     print(f"Split document into {len(chunks)} chunks.")

#     # 4. Initialize Embeddings and PGVector Database
#     print("Initializing OpenAI Embeddings...")
#     embeddings = OpenAIEmbeddings()
    
#     print("Connecting to PGVector Database...")
#     db = PGVector(
#         connection=DATABASE_URL,
#         embeddings=embeddings,
#         collection_name="financial_reports",
#         use_jsonb=True,
#     )
    
#     # 5. Push Data to Database
#     print("Uploading chunks to the vector database... (This may take a minute)")
#     db.add_documents(chunks)
#     print("✅ Ingestion complete! The new SEC data is now live in your database.")
def fetch_and_embed_latest_10q(ticker="AAPL"):
    # 1. Authenticate with the SEC (Legally required)
    print("Connecting to SEC EDGAR...")
    set_identity("Kavan Patel kavanpatel2834@example.com") 
    
    # 2. Download the filing
    company = Company(ticker)
    filing = company.get_filings(form="10-Q")[0]
    print(f"Downloaded {ticker} 10-Q filed on {filing.filing_date}")
    
    # 3. Extract clean text
    filing_obj = filing.obj()
    text = filing_obj.text()
    
    # 4. Chunk the text
    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # 5. Create Documents and unique IDs for Upserting (Task 2)
    documents = []
    ids = []
    for i, chunk in enumerate(chunks):
        # Creating a unique ID prevents duplicates if the script runs twice
        doc_id = f"{ticker}_{filing.accession_no}_chunk_{i}"
        
        doc = Document(
            page_content=chunk, 
            metadata={
                "source": "SEC", 
                "ticker": ticker, 
                "date": filing.filing_date
            }
        )
        documents.append(doc)
        ids.append(doc_id)
        
    print(f"Upserting {len(documents)} chunks into PGVector...")
    
    # 6. Database Upsert
    embeddings = OpenAIEmbeddings()
    db = PGVector(
        connection_string=DATABASE_URL,
        embedding_function=embeddings,
        collection_name="financial_docs" # Change this if your collection is named differently
    )
    
    # Passing ids to add_documents ensures it updates existing rows instead of duplicating
    db.add_documents(documents, ids=ids)
    print("Success! Database is up to date.")

if __name__ == "__main__":
    # You can easily change this to TSLA, MSFT, or NVDA
    # fetch_and_ingest_sec_filing("AAPL", "10-Q")
    fetch_and_embed_latest_10q("AAPL")