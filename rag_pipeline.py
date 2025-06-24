import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from pdf_chunker import parse_and_chunk_pdfs

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("🛠️ Loaded key (first 10 chars):", OPENAI_API_KEY[:10])

# Initialize embedding model
try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
except Exception as e:
    print("❌ Error initializing OpenAIEmbeddings:", e)
    raise

def create_faiss_index(chunks):
    """
    Build and save a FAISS index from provided text chunks.
    """
    docs = [Document(page_content=chunk) for chunk in chunks]
    print("🔄 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index saved to /faiss_index")


def load_faiss_index():
    """
    Load an existing FAISS index or rebuild it if missing or corrupted.
    """
    index_path = "faiss_index"
    # Rebuild if missing
    if not os.path.exists(index_path):
        print("⚠️ FAISS index not found — rebuilding from PDF.")
        chunks = parse_and_chunk_pdfs()
        if not chunks:
            raise ValueError("No chunks returned from parse_and_chunk_pdfs()")
        create_faiss_index(chunks)
    else:
        print("✅ FAISS index found. Attempting to load...")

    # Attempt to load, rebuild on failure
    try:
        return FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print("❌ Failed to load FAISS index (corrupted or incompatible), rebuilding...", e)
        chunks = parse_and_chunk_pdfs()
        if not chunks:
            raise ValueError("No chunks to rebuild FAISS index")
        create_faiss_index(chunks)
        return FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )


def query_vectorstore(vectorstore, query, k=3):
    """
    Perform a similarity search on the vectorstore.
    Returns the top-k page contents.
    """
    print(f"🔍 Searching for: '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]


if __name__ == "__main__":
    chunks = parse_and_chunk_pdfs()
    create_faiss_index(chunks)

    db = load_faiss_index()
    answers = query_vectorstore(db, "How can I motivate an avoidantly attached employee?", k=3)
    for i, a in enumerate(answers, 1):
        print(f"\n--- Match {i} ---\n{a[:500]}...")
