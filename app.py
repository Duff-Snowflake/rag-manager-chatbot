import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from rag_pipeline import load_faiss_index
from langchain.chains import RetrievalQA
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

USER_DB_PATH = "user_access.json"
EMAIL_DB_PATH = "user_access.json"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UNRESTRICTED_EMAIL = "duffwarrenconsulting@gmail.com"
REQUIRED_PASSWORD = "b@6J8KJNff9*&^N:ll3Fb@r@3"
ACCESS_DURATION_DAYS = 7

retriever = None
qa_chain = None
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach with deep expertise in attachment theory and interpersonal motivation.

Imagine the following manager has asked you a question:

QUERY: {query}
ANSWER: {base_answer}

Respond as if you're speaking directly to the manager in a conversational, coaching style.

1. Begin with a short, encouraging summary of your interpretation of their situation and your high-level advice — spoken as if you're in a 1-on-1 session.
2. Then, naturally introduce 6 specific example phrases the manager could say in this situation. For each one, briefly explain *why* it works, grounded in psychological principles — but keep the tone human, not clinical.
3. End with a short reflective question to prompt the manager to consider how they might apply these suggestions with their specific employee.
4. Write the full response as markdown, in a voice that feels like a warm, confident expert guiding someone through a challenge.

Avoid repeating the query or answer unless it's helpful to reframe. Be clear, empathetic, and concrete.
"""
    return llm.invoke(prompt).content

if os.path.exists(EMAIL_DB_PATH):
    with open(EMAIL_DB_PATH, "r") as f:
        access_db = json.load(f)
else:
    access_db = {}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = ""

if not st.session_state.authenticated:
    st.markdown("### Understanding and motivating your teams")
    st.markdown("##### Personality types and how to get the most out of them")
    email_input = st.text_input("Please enter your email to access the assistant:", value="", max_chars=100)
    if email_input:
        st.session_state.email = email_input
    email = st.session_state.email

    if email == UNRESTRICTED_EMAIL:
        password = st.text_input("Enter your password:", type="password")
        if password:
            if password == REQUIRED_PASSWORD:
                st.success("Admin access granted.")
                st.session_state.authenticated = True
                view_as_user = st.checkbox("View as regular user")
                if view_as_user:
                    expiry = datetime.now() + timedelta(days=ACCESS_DURATION_DAYS)
                    st.success(f"Simulated user access. Trial active until {expiry.date()}")
                else:
                    st.markdown("You are viewing full admin capabilities.")
            else:
                st.error("Incorrect password.")
                st.stop()
        else:
            st.stop()
    elif email:
        if email not in access_db:
            access_db[email] = (datetime.now() + timedelta(days=ACCESS_DURATION_DAYS)).isoformat()
            with open(EMAIL_DB_PATH, "w") as f:
                json.dump(access_db, f)
        expiry = datetime.fromisoformat(access_db.get(email, "2000-01-01T00:00:00"))
        if datetime.now() > expiry:
            st.error("❌ Your trial has expired. Please contact us to extend access.")
            st.stop()
        else:
            st.success(f"Access granted. Trial active until {expiry.date()}")
            st.session_state.authenticated = True
    else:
        st.warning("Please enter your email to begin your free trial.")
        st.stop()

if st.session_state.authenticated:
    email = st.session_state.email
    st.success(f"Logged in as: {email}")
    col1, col2, col3 = st.columns([5, 2, 2])
    with col3:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.email = ""
            st.rerun()

    if retriever is None:
        retriever = load_faiss_index().as_retriever(return_source_documents=True)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # CSS and Logo Banner here (shortened for brevity)
    st.markdown("""<style>/* STYLES INSERTED HERE */</style>""", unsafe_allow_html=True)
    st.markdown("""<div>...logo...</div>""", unsafe_allow_html=True)

    st.markdown("I am an agent designed to help you better motivate your team and direct reports.")
    st.markdown("The language we use directly affects how people respond to us. Mastering this creates productive, deadline-hitting teams.")

    if "query" not in st.session_state:
        st.session_state.query = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input field
    user_input = st.text_input("Or enter your question here", value=st.session_state.query, placeholder="e.g., How do I give feedback to an avoidant employee?")
    st.session_state.query = user_input

    # Chat response (streamed above buttons)
    if st.session_state.query and qa_chain:
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

    # Centered clear button
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("Clear Response History"):
        st.session_state.history = []
    st.markdown('</div>', unsafe_allow_html=True)

    # Example questions
    with st.expander("Questions to get you started", expanded=False):
        st.markdown('<div class="question-buttons">', unsafe_allow_html=True)
        example_questions = [
            "How can I figure out what type of person I am dealing with?",
            "How do I motivate someone with an anxious attachment style?",
            "How do I give feedback to an avoidant employee?",
            "How can I deliver bad news without making someone shut down?",
            "What should I say when an employee takes credit for others' work?"
        ]
        for i, q in enumerate(example_questions):
            st.markdown('<div class="button-wrapper">', unsafe_allow_html=True)
            if st.button(q, key=f"example_{i}"):
                st.session_state.query = q
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer logo
    st.markdown("""<div class="footer-logo"><img src="...logo path..." alt="Your Branding Here"></div>""", unsafe_allow_html=True)
