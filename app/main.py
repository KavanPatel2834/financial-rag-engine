from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import PGVector

from app.database.config import DATABASE_URL

load_dotenv()

app = FastAPI(
    title="Financial RAG Engine",
    description="A Retrieval-Augmented Generation (RAG) engine for financial data using LangChain and OpenAI.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list[str]

@app.get("/")
async def health_check():
    return {"status": "Financial RAG Engine is running."}

# --- RAG Pipeline Initialization ---
# We initialize this outside the endpoint so the server doesn't have to 
# reconnect to the database and re-download the model on every single request.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = PGVector(
    connection_string=DATABASE_URL,
    embedding_function=embeddings,
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

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# -----------------------------------

@app.post("/query",response_model=QueryResponse)
async def ask_finacial_question(request: QueryRequest):
    try:
        # # Pass the user's question from the API request to the RAG pipeline
        # response = rag_chain.invoke(request.question)
        # return QueryResponse(answer=response)

        #Step 1: Manually trigger the database search and save the documents
        docs = retriever.invoke(request.question)   
        
        #Step 2: Manually trigger the prompt and LLM generation using the retrieved documents
        context_text = "\n\n".join(doc.page_content for doc in docs)
        
        #Step 3: Create the prompt with the context and question
        answer = generation_chain.invoke({
            "context": context_text,
              "question": request.question
              })
        
        #Extract the raw text from the answer and return it along with the source documents
        sources = [doc.page_content for doc in docs]

        # Step 5: Package it all up in our strict pydantic JSON structure
        return QueryResponse(answer=answer, source_documents=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))