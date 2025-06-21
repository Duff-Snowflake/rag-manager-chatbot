import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from rag_pipeline import load_faiss_index
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load retriever
retriever = load_faiss_index().as_retriever(return_source_documents=True)

# Load LLM and QA chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Inject custom CSS for dark corporate theme
st.markdown("""
    <style>
    html, body {
        background-color: #1b2a41;
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .main {
        background-color: #27374d;
        color: #ffffff;
        padding: 2rem;
        padding-bottom: 6rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        max-width: 800px;
        margin: auto;
        max-height: 90vh;
        overflow-y: auto;    
    }
    .stTextInput > div > div > input {
        background-color: #324a63;
        color: #ffffff;
        font-size: 1.3rem;
        padding: 0.75rem;
        border-radius: 6px;
        border: none;
    }
    .stSpinner {
        color: #ffffff !important;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
    }
    header, .block-container:has(header), .css-1avcm0n.ezrtsby2 {
        display: none !important;
    }
    footer {
        visibility: hidden;
    }
    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button {
        width: 70%;
        text-align: center;
        padding: 0.75rem;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        background-color: #324a63;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .footer-logo {
        text-align: center;
        margin-top: 2rem;
    }
    .footer-logo img {
        width: 150px;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# Layout
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Header branding with title inside the box
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Logo" style="height: 50px; margin-right: 10px;">
    <h2 style="color: white; margin: 0;">Employee Management Assistant</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("##### Example Questions")

example_questions = [
    "How can I figure out what type of person I am dealing with?",
    "How do I motivate someone with an anxious attachment style?",
    "How do I give feedback to an avoidant employee?",
    "How can I deliver bad news without making someone shut down?",
    "What should I say when an employee takes credit for others' work?" 
]

if "query" not in st.session_state:
    st.session_state.query = ""

if "history" not in st.session_state:
    st.session_state.history = []

# Question buttons
for i, q in enumerate(example_questions):
    if st.button(q, key=f"example_{i}"):
        st.session_state.query = q

# Input box
user_input = st.text_input("Or enter your question here", value=st.session_state.query, placeholder="e.g., How do I give feedback to an avoidant employee?")
st.session_state.query = user_input

# History controls
if st.button("Clear Response History"):
    st.session_state.history = []

show_history = st.checkbox("Show response history")
show_sources = st.checkbox("Show source documents")

def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach trained in attachment theory.
Based on the following manager query and response:

QUERY: {query}
ANSWER: {base_answer}

Please output the following format:

1. A refined and professional version of the answer above.
2. Then, a list of 4 to 8 concrete example phrases the manager could say.
   For each example, include a one-sentence explanation of *why* it works (the psychological or relational principle it supports).
Output everything as markdown.
"""
    return llm.invoke(prompt).content

if st.session_state.query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": st.session_state.query})
        base_answer = result["result"]
        formatted = format_response(base_answer, st.session_state.query)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "q": st.session_state.query,
            "a": formatted,
            "t": timestamp,
            "sources": result.get("source_documents", [])
        })
    st.markdown(formatted)
    if show_sources:
        st.markdown("**Sources:**")
        for doc in result.get("source_documents", []):
            st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")

if show_history and st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🔁 Previous Responses")
    for entry in st.session_state.history:
        st.markdown(f"**Q ({entry['t']}):** {entry['q']}")
        st.markdown(entry['a'])
        if show_sources and entry["sources"]:
            st.markdown("**Sources:**")
            for doc in entry["sources"]:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")
        st.markdown("---")

# Footer branding
st.markdown("""
<div class="footer-logo">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Your Branding Here">
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
