from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.database.config import DATABASE_URL

load_dotenv()

def run_rag_pipeline():
    print("1. Connecting to the vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = PGVector(
        connection=DATABASE_URL,
        embeddings=embeddings,
        collection_name="financial_reports",
        use_jsonb=True,
    )

    # Configure the database to act as a retriever tool fetching the top 4 chunks
    retriever = db.as_retriever(search_kwargs={"k": 4})

    print("2. Intializing GPT-4o-mini...")
    # Initialize the GPT-4o-mini model with a temperature of 0 for deterministic responses
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    #Define the Augmentation prompt
    template = """You are a helpful finacial assistant. Answer the question using ONLY the following context.
    If you don't know the answer based on the context, just say that you don't know. 
    
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    #Helper function to merge the text from multiple database chunks
    def formate_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the data pipeline using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | formate_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )

    query = "How much revenue did Apple make in Q3?"
    print(f"\n3. Executing pipeline for:'{query}'")
    print("Thinking...\n")

    #Runt hte query through the entire retrieval and generation chain
    response = rag_chain.invoke(query)

    print("✅ --- Final LLM Answer ---")
    print(response)

if __name__ == "__main__":
    run_rag_pipeline()