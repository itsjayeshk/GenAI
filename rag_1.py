from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pdf_path = Path(__file__).parent / "resume.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

print("PDF loaded successfully")
print("Number of pages:", len(docs))


# --------------------------------------------------
# 2. Split PDF into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(docs)

print("PDF split successfully")
print("Number of chunks:", len(split_docs))


# --------------------------------------------------
# 3. Create Gemini embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("Gemini embedding model initialized")


# --------------------------------------------------
# 4. Store documents in Qdrant
# --------------------------------------------------

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="learning_langchain"
)

print("Documents successfully stored in Qdrant")


# --------------------------------------------------
# 5. Connect to existing Qdrant collection
# --------------------------------------------------

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings
)

print("Connected to Qdrant collection")


# --------------------------------------------------
# 6. User Question
# --------------------------------------------------

question = "What is the year of the student?"


# --------------------------------------------------
# 7. Similarity Search
# --------------------------------------------------

search_result = retriever.similarity_search(
    query=question,
    k=3
)

print("\nRelevant documents retrieved:", len(search_result))


# --------------------------------------------------
# 8. Create relevant context
# --------------------------------------------------

relevant_chunks = "\n\n".join(
    result.page_content
    for result in search_result
)


# --------------------------------------------------
# 9. Create Prompt
# --------------------------------------------------

SYSTEM_PROMPT = f"""
You are a helpful AI assistant who answers questions
based only on the provided context.

If the answer is not present in the context,
say that you don't know.

Context:
{relevant_chunks}

Question:
{question}
"""


# --------------------------------------------------
# 10. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)


# --------------------------------------------------
# 11. Generate Answer
# --------------------------------------------------

response = llm.invoke(SYSTEM_PROMPT)


# --------------------------------------------------
# 12. Display Answer
# --------------------------------------------------

print("\n================================")
print("ANSWER")
print("================================")

print(response.content)

print("================================")