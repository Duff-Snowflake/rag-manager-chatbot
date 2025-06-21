import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from pdf_chunker import parse_and_chunk_pdfs

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("🛠️ Loaded key (first 10 chars):", OPENAI_API_KEY[:10])

# Create embeddings using OpenAI
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

def create_faiss_index(chunks):
    # Wrap text chunks in Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    print("🔄 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Save it locally
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index saved to /faiss_index")

def load_faiss_index():
    print("📂 Loading FAISS index...")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

def query_vectorstore(vectorstore, query, k=3):
    print(f"\n🔍 Searching for: '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]

if __name__ == "__main__":
    chunks = parse_and_chunk_pdfs()
    create_faiss_index(chunks)

    # Example query:
    db = load_faiss_index()
    answers = query_vectorstore(db, "How can I motivate an avoidantly attached employee?")
    
    for i, a in enumerate(answers, 1):
        print(f"\n--- Match {i} ---\n{a[:500]}...")
