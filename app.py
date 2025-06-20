import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from rag_pipeline import load_faiss_index
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load retriever
retriever = load_faiss_index().as_retriever()

# Load LLM and QA chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

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
        padding-bottom: 4rem;
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
        font-size: 1.1rem;
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
    </style>
""", unsafe_allow_html=True)

# Layout
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; margin-bottom: 1rem;'>Employee Management Assistant</h1>", unsafe_allow_html=True)

st.markdown("##### Example Questions")

# Define example questions
example_questions = [
    "How can I figure out what type of person I am dealing with?",
    "How do I motivate someone with an anxious attachment style?",
    "How do I give feedback to an avoidant employee?",
    "How can I deliver bad news without making someone shut down?",
    "What should I say when an employee takes credit for others' work?" 
]

# Session state for question management
if "query" not in st.session_state:
    st.session_state.query = ""

# Button-driven input setter
for i, q in enumerate(example_questions):
    if st.button(q, key=f"example_{i}"):
        st.session_state.query = q

# Text input always visible
user_input = st.text_input("Or enter your question here", value=st.session_state.query, placeholder="e.g., How do I give feedback to an avoidant employee?")

# Update query from text input
st.session_state.query = user_input

def format_response(base_answer):
    prompt = f"""
You are a management communication coach trained in attachment theory.
Based on the following manager query and response:

QUERY: {st.session_state.query}
ANSWER: {base_answer}

Please output the following format:

1. A refined and professional version of the answer above.
2. Then, a list of 6 concrete example phrases the manager could say.
   For each example, include a one-sentence explanation of *why* it works (the psychological or relational principle it supports).
Output everything as markdown.
"""
    return llm.invoke(prompt).content

# If query is submitted
if st.session_state.query:
    with st.spinner("Thinking..."):
        base_answer = qa_chain.run(st.session_state.query)
        formatted = format_response(base_answer)
    st.markdown(formatted)

st.markdown("</div>", unsafe_allow_html=True)
