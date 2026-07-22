from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.database.config import DATABASE_URL

load_dotenv()

def run_test_query():
    print("1. Connecting to the vector store...")
    # Initialize the OpenAI embeddings model
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # Connect to the PostgreSQL vector store
    db = PGVector(
        collection_name="financial_reports",
        connection=DATABASE_URL,
        embeddings=embedding,
    )

    # Perform a test query
    query = "How much revenue did Apple generate in Q3?"

    # Perform a similarity search in the vector store to find the top 2 most relevant document chunks related to the query
    results = db.similarity_search(query, k=4)

    print("\n✅ --- Top Results Found ---")
    for i,doc in (enumerate(results)):
        print(f"\nResult {i+1}:")
        print(doc.page_content)

if __name__ == "__main__":
    run_test_query()
      
