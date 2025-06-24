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

    st.markdown("""
        <style>
        body, .main, .block-container {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

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
        }

        .chat-box {
            max-height: 400px;
            overflow-y: auto;
            background-color: #1e2e3f;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #324a63;
        }
                
        .chat-entry {
            margin-bottom: 1.5rem;
        }
                
        .chat-question {
            font-weight: bold;
            margin-bottom: 0.25rem;
            color: #ffffff;
        }
                
        .chat-response {
            color: #dddddd;
            white-space: pre-wrap;
        }      

        .question-buttons {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }

        .button-wrapper {
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .centered-button {
            display: flex;
            justify-content: center;
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
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        footer {
            visibility: hidden;
        }

        div.stButton {
            display: flex;
            justify-content: center;
        }

        div.stButton > button {
            white-space: nowrap;
            width: auto !important;
            padding: 0.5rem 1rem;
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

    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Logo" style="height: 50px; margin-right: 10px;">
        <h2 style="color: white; margin: 0;">Employee Management Assistant</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("I am an agent that is designed to help you with understanding how to better motivate the members of your teams and your direct reports.")

    st.markdown("The language we use can directly affect how people respond to us.  Mastering this allows us to create more productive teams that meet and exceed deadlines, produce quality and have lower levels of passive push-back.")

    # Scrollable chat window
    if "history" in st.session_state and st.session_state.history:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for entry in st.session_state.history:
            st.markdown(f'''
                <div class="chat-entry">
                    <div class="chat-question">Q ({entry["t"]}): {entry["q"]}</div>
                    <div class="chat-response">{entry["a"]}</div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input(
        "Or enter your question here",
        value=st.session_state.query,
        placeholder="e.g., How do I give feedback to an avoidant employee?"
    )
    st.session_state.query = user_input

    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("Clear Response History"):
        st.session_state.history = []
    st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown("""
    <div class="footer-logo">
        <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Your Branding Here">
    </div>
    """, unsafe_allow_html=True)
