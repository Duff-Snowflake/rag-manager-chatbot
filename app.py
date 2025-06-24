import streamlit as st
import streamlit.components.v1 as components
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
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Process submitted query before UI renders
if "submitted_query" in st.session_state and st.session_state.submitted_query and qa_chain:
    query = st.session_state.submitted_query
    st.session_state.submitted_query = ""  # Clear early to prevent loops
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        base_answer = result["result"]
        formatted = format_response(base_answer, query)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "q": query,
            "a": formatted,
            "t": timestamp,
            "sources": result.get("source_documents", [])
        })
    st.rerun()

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

    st.markdown("""
        <style>
        .chat-box {
            max-height: 400px;
            overflow-y: auto;
            background-color: #1e2e3f;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #324a63;
        }
        .chat-entry { margin-bottom: 1.5rem; }
        .chat-question { font-weight: bold; color: #fff; }
        .chat-response { color: #ddd; white-space: pre-wrap; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style=\"display: flex; align-items: center; margin-bottom: 1rem;\">
        <img src=\"https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png\" alt=\"Logo\" style=\"height: 50px; margin-right: 10px;\">
        <h2 style=\"color: white; margin: 0;\">Employee Management Assistant</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("I am an agent that is designed to help you with understanding how to better motivate the members of your teams and your direct reports.")
    st.markdown("The language we use can directly affect how people respond to us. Mastering this allows us to create more productive teams.")

    with st.expander("Some sample questions to get you started", expanded=False):
        example_questions = [
            "How can I figure out what type of person I am dealing with?",
            "How do I motivate someone with an anxious attachment style?",
            "How do I give feedback to an avoidant employee?",
            "How can I deliver bad news without making someone shut down?",
            "What should I say when an employee takes credit for others' work?"
        ]
        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}"):
                st.session_state.submitted_query = q

    col_submit, col_clear = st.columns([1, 1])

    with col_submit:
        query_input = st.text_input(
            "Talk to me about what you are having trouble with",
            key="query_input_box",
            placeholder="e.g., How do I give feedback to an avoidant employee?"
        )
        if st.button("Submit", key="submit_button"):
            if query_input and (not st.session_state.history or query_input != st.session_state.history[-1]["q"]):
                st.session_state.submitted_query = query_input

    with col_clear:
        if st.button("Clear Response History", key="clear_button"):
            st.session_state.history = []

    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for entry in st.session_state.history:
            st.markdown(f'<div class="chat-entry">', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-question">👤 {entry["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-response">{entry["a"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
            const chatBox = window.parent.document.querySelector('.chat-box');
            if (chatBox) {
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            </script>
            """,
            height=0,
            scrolling=False
        )

    st.markdown("""
    <div class="footer-logo">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Your Branding Here" style="width: 0.25%;">
</div>
    """, unsafe_allow_html=True)
