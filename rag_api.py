from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG once on startup
docs_dir = Path("docs")
documents = []
for doc_file in docs_dir.glob("*.txt"):
    loader = TextLoader(str(doc_file))
    documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="mistral")

# FastAPI app
app = FastAPI(title="Iris RAG Chatbot", version="1.0")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask(query: Query):
    """Ask the RAG system about iris flowers."""
    docs = retriever.invoke(query.question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = (
        f"Based on this context: {context}\n\nQuestion: {query.question}\n\nAnswer:"
    )
    answer = llm.invoke(prompt)

    logger.info(f"Question: {query.question}")
    logger.info(f"Answer: {answer}")

    return {"question": query.question, "answer": answer, "source_documents": len(docs)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
