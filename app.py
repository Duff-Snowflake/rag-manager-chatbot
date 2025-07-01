import streamlit as st
st.set_page_config(page_title="Employee Management Assistant", layout="centered")
import streamlit.components.v1 as components
from langchain_community.chat_models import ChatOpenAI
from rag_pipeline import load_faiss_index
from langchain.chains import RetrievalQA
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment and initialize LLM
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach with deep expertise in attachment theory and interpersonal motivation.

Imagine the following manager has asked you a question:

QUERY: {query}
ANSWER: {base_answer}

Respond as if you're speaking directly to the manager in a conversational, coaching style.

1. Begin with a short, encouraging summary of your interpretation of their situation and your high-level advice — spoken as if you're in a 1-on-1 session.
2. Do not start your encouraging summary of your interpretation of their situation with "Absolutely".
3. Then, naturally introduce 4 specific example phrases the manager could say in this situation that are spectful and direct. Use language appropriate to anxious or avoidant personalities. 
4. For each one, briefly explain *why* it works, grounded in psychological principles — but keep the tone human, not clinical.
5. End with a short reflective question to prompt the manager to consider how they might apply these suggestions with their specific employee.
6. Write the full response as markdown, in a voice that feels like a warm, confident expert helping someone understand differnent personlity types and how to motivate them.

Avoid repeating the query or answer unless it's helpful to reframe. Be clear, empathetic, and concrete.
"""
    return llm.invoke(prompt).content

# Constants
UNRESTRICTED_EMAIL = "duffwarrenconsulting@gmail.com"
REQUIRED_PASSWORD = "b@6J8KJNff9*&^N:ll3Fb@r@3"
ACCESS_DURATION_DAYS = 7

# Initialize Streamlit session state variables
for key, default in {
    "authenticated": False,
    "email": "",
    "submitted_query": "",
    "history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Global CSS styling
st.markdown("""
<style>
body { background-color: #343541; color: white; }

/* Container styling */
.chat-box {
    height: 400px;  /* Adjust height as needed */
    overflow-y: auto;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    background: #2f3136;
    border: 1px solid #565869;
}

/* Individual message entry */
.chat-entry {
    margin-bottom: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

/* User question bubble */
.chat-question {
    align-self: flex-end;
    background-color: #40414f;
    color: #fff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Assistant response bubble */
.chat-response {
    align-self: flex-start;
    background-color: #444654;
    color: #fff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    white-space: pre-wrap;
    line-height: 1.6;  /* Adjust to your preferred spacing */
}

/* Format numbered lists for tight spacing */
.chat-response ol {
    padding-left: 1.2rem;  /* Adjust left indent of ordered list */
}

.chat-response ol li {
    margin-bottom: 0.3rem; /* Reduce spacing between numbered items */
}

/* Tighten suggestion and explanation spacing if using paragraphs */
.chat-response p {
    margin-bottom: 0.3rem;
}
            
/* Input field styling */
input[type="text"] {
    border-radius: 8px !important;
    padding: 0.75rem !important;
    background-color: #40414f !important;
    color: white !important;
    border: 1px solid #565869 !important;
}

/* Button styling */
button[kind="primary"] {
    background-color: #40414f !important;
    color: white !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #4e4f5c !important;
}
</style>
""", unsafe_allow_html=True)

# Authentication UI
if not st.session_state.authenticated:
    st.markdown("### Understanding and motivating your teams")
    st.markdown("##### Personality types and how to get the most out of them")
    st.markdown("The types of people in our workplaces are more varied now than ever. More people than ever are more sensitive to criticism and need more encouragement in order to maintain engagement and productivity. This app is your coach to understanding these new needs and being able to leverage your teams, just by learning how to talk to different people.")

    email_input = st.text_input("Enter your email to access the assistant:", max_chars=100)
    if email_input:
        st.session_state.email = email_input

    if st.session_state.email == UNRESTRICTED_EMAIL:
        pwd = st.text_input("Admin password:", type="password")
        if pwd == REQUIRED_PASSWORD:
            st.success("Admin access granted.")
            st.session_state.authenticated = True
        elif pwd:
            st.error("Incorrect password.")
    elif st.session_state.email:
        # load or create trial expiry
        db = {}
        try:
            with open("user_access.json", "r") as f:
                db = json.load(f)
        except FileNotFoundError:
            pass

        expiry = db.get(st.session_state.email)
        if not expiry:
            db[st.session_state.email] = (datetime.now() + timedelta(days=ACCESS_DURATION_DAYS)).isoformat()
            with open("user_access.json", "w") as f:
                json.dump(db, f)
            expiry = db[st.session_state.email]

        if datetime.now() > datetime.fromisoformat(expiry):
            st.error("❌ Trial expired. Contact us to extend access.")
        else:
            st.success(f"Access granted until {datetime.fromisoformat(expiry).date()}")
            st.session_state.authenticated = True

    if not st.session_state.authenticated:
        st.stop()

# Authenticated users skip the block above and app continues below

# Initialize retriever and QA chain
try:
    retriever = load_faiss_index().as_retriever(return_source_documents=True)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
except Exception as e:
    st.error(f"Error loading FAISS index or setting up RetrievalQA: {e}")
    st.stop()

# Process any new submitted query
to_query = st.session_state.submitted_query
if to_query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": to_query})
        formatted = format_response(result["result"], to_query)
        st.session_state.history.insert(0, {
            "q": to_query,
            "a": formatted,
            "t": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": result.get("source_documents", [])
        })
    st.session_state.submitted_query = ""
    st.success("Response added. Scroll down to view it.")

# Page header
st.success(f"Logged in as: {st.session_state.email}")
if st.button("Logout", key="logout"):
    st.session_state.authenticated = False
    st.session_state.email = ""
    st.experimental_rerun()

# Sample questions
with st.expander("Sample questions to get you started", expanded=False):
    for i, q in enumerate([
        "How do I motivate an anxious type?",
        "How do I motivate an avoidant type?",
        "How to give feedback to an employee who dodges accountability?", 
        "How do I deliver bad news without shutting someone down?"
    ]):
        if st.button(q, key=f"example_{i}"):
            st.session_state.submitted_query = q

# Chat history display
st.markdown('<div class="chat-box">', unsafe_allow_html=True)

# ✅ Added placeholder if no history
if not st.session_state.history:
    st.markdown('<div style="color: #888;">No conversation yet. Ask a question below!</div>', unsafe_allow_html=True)
else:
    for i, entry in enumerate(st.session_state.history):
        st.markdown(f'''
            <div class="chat-entry">
                <div class="chat-question">{entry["q"]}</div>
                <div class="chat-response">{entry["a"]}</div>
            </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input form at bottom
with st.form("query_form"):
    query = st.text_input("Your question:", placeholder="Type your question and press Enter or Submit")
    submitted = st.form_submit_button("Submit")
    if submitted and query.strip():
        st.session_state.submitted_query = query.strip()

if st.button("Clear History", key="clear_button"):
    st.session_state.history = []

components.html("""
<script>
const chatBox = window.parent.document.querySelector('.chat-box');
if(chatBox){ chatBox.scrollTop = chatBox.scrollHeight; }
</script>
""", height=0)

# Footer logo
st.markdown("""
<div style='text-align:center; margin-top:2rem;'>
  <img src='https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png' alt='Brand' style='width:10px;'>
</div>
""", unsafe_allow_html=True)
