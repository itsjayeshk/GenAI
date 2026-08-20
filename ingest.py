from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

pdf_path = Path(__file__).parent / "resume.pdf"
docs = PyPDFLoader(file_path=pdf_path).load()

split_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="learning_langchain"
)

print(f"Ingested {len(split_docs)} chunks into Qdrant")