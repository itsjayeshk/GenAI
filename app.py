import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Resume Q&A", page_icon="📄")


# --------------------------------------------------
# Cached resources (created once, reused across reruns)
# --------------------------------------------------

@st.cache_resource
def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embeddings
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )


# --------------------------------------------------
# Chat UI
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 Resume Q&A Bot")
st.caption("Answers are generated only from the resume document.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new question
if question := st.chat_input("Ask a question about the resume..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Retrieve relevant chunks
        with st.spinner("Searching the resume..."):
            search_result = get_retriever().similarity_search(query=question, k=3)
            relevant_chunks = "\n\n".join(
                doc.page_content for doc in search_result
            )

        prompt = f"""You are a helpful AI assistant who answers questions
based only on the provided context.

If the answer is not present in the context, say that you don't know.

Context:
{relevant_chunks}

Question:
{question}"""

        # Stream the answer token by token
        response_text = st.write_stream(get_llm().stream(prompt))

        # Show which chunks were used
        with st.expander("📚 View source chunks"):
            for i, doc in enumerate(search_result, 1):
                st.markdown(f"**Chunk {i}** — Page {doc.metadata.get('page', '?') + 1}")
                st.text(doc.page_content)
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )