# Financial RAG Engine

An AI-powered API that answers financial questions about public companies using data sourced directly from SEC EDGAR filings. This engine uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide accurate, evidence-backed answers.

## ✨ Features

*   **Automated SEC Data Ingestion**: Automatically downloads and processes the latest 10-Q quarterly filings for a given company ticker (e.g., "AAPL").
*   **Layout-Aware Chunking**: Uses `MarkdownHeaderTextSplitter` to intelligently chunk documents based on their structural headers, keeping related financial data and tables intact.
*   **Hybrid Search**: Implements a powerful hybrid search strategy by combining:
    *   **Dense (Vector) Search**: For finding semantically similar text.
    *   **Sparse (Keyword) Search**: Using BM25 for matching specific terms and numbers.
*   **Reciprocal Rank Fusion (RRF)**: Merges the results from both search methods to produce a single, highly relevant list of source documents.
*   **LLM-Powered Generation**: Leverages `gpt-4o-mini` to generate a final, coherent answer based on the retrieved context.
*   **Duplicate Prevention**: Uses an `UPSERT` mechanism with unique document IDs to ensure that running the ingestion pipeline multiple times does not create duplicate entries in the vector database.
*   **Scheduled Daily Updates**: A background scheduler runs daily to automatically fetch the latest filings, ensuring the database remains current.
*   **FastAPI Interface**: Exposes a clean, simple `/query` endpoint to interact with the RAG engine.

## 🚀 Getting Started

### Prerequisites

*   Docker and Docker Compose
*   Python 3.9+
*   An OpenAI API key

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd financial-rag-engine
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key and SEC identity. The `DATABASE_URL` is configured for the Docker setup.

    ```env
    OPENAI_API_KEY="sk-..."

    # The SEC requires a User-Agent for API requests.
    # Replace with your own name and email.
    SEC_IDENTITY="Your Name your.email@example.com"

    # Default connection string for the Dockerized PostgreSQL database
    DATABASE_URL="postgresql+psycopg://user:password@db:5432/vectordb"
    ```

3.  **Build and run the services:**
    ```bash
    docker-compose up --build -d
    ```
    This will start the FastAPI application, the PostgreSQL database with PGVector, and the background scheduler.

4.  **Perform the initial data ingestion:**
    Run the ingestion script to populate the database with the first SEC filing.

    ```bash
    docker-compose exec api python -m app.database.ingest
    ```

### Usage

You can ask the engine a financial question by sending a POST request to the `/query` endpoint.

```bash
curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "How much revenue did Apple generate in the last quarter?"
}'
```