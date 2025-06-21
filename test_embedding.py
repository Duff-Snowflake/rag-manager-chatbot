from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
print("🔑 Using key (first 10 chars):", key[:10])

embeddings = OpenAIEmbeddings(openai_api_key=key)
result = embeddings.embed_query("test this works")
print("✅ Embedding result (first 5 values):", result[:5])
