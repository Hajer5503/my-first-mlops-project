from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path

# Load documents
docs_dir = Path("docs")
documents = []
for doc_file in docs_dir.glob("*.txt"):
    loader = TextLoader(str(doc_file))
    documents.extend(loader.load())

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Create embeddings with nomic-embed-text and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create RAG with LLM
llm = OllamaLLM(model="mistral")
retriever = vectorstore.as_retriever()

# Test RAG
if __name__ == "__main__":
    queries = [
    "What are the measurements of a virginica iris?",
    "Which iris species has the largest petal length?",
    "What is the petal width range for versicolor?"
]

for query in queries:
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Based on this context: {context}\n\nAnswer: {query}"
    result = llm.invoke(prompt)
    print(f"Query: {query}")
    print(f"Answer: {result}\n")
    