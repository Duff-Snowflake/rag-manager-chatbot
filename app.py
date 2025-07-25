import streamlit as st
import os
import json
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from rag_pipeline import load_faiss_index

# Streamlit config
st.set_page_config(page_title="Employee Management Assistant", layout="centered")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")
REQUIRED_PASSWORD = os.getenv("REQUIRED_PASSWORD")

# Constants
UNRESTRICTED_EMAIL = "duffwarrenconsulting@gmail.com"
ACCESS_DURATION_DAYS = 7

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Session state init
for key, default in {
    "authenticated": False,
    "email": "",
    "submitted_query": "",
    "history": [],
    "video_url": None,
    "latest_question": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load access database
db = {}
try:
    with open("user_access.json", "r") as f:
        db = json.load(f)
except FileNotFoundError:
    pass

# Authentication
if not st.session_state.authenticated:
    st.markdown("### Understanding and motivating your teams")
    st.markdown("##### Personality types and how to get the most out of them")
    st.markdown("This app is your coach for learning how to talk to different people and motivate your teams.")

    email_input = st.text_input("Enter your email to access the assistant:", max_chars=100)
    if email_input:
        st.session_state.email = email_input

    if st.session_state.email == UNRESTRICTED_EMAIL:
        pwd = st.text_input("Admin password:", type="password")
        if pwd == REQUIRED_PASSWORD:
            st.success("Admin access granted.")
            st.session_state.authenticated = True
            st.rerun()
        elif pwd:
            st.error("Incorrect password.")

    elif st.session_state.email:
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
            st.rerun()

    if not st.session_state.authenticated:
        st.stop()

# Extended session keys post-login
for key, default in {
    "video_url": None,
    "latest_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Display banner
st.markdown(f"""
<div class="top-banner">
  <div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Logo" height="30">
    <strong>Employee Management Assistant</strong>
  </div>
  <div class="status">
    Logged in as: {st.session_state.email}
  </div>
</div>
<style>
.top-banner {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #202123;
    color: white;
    padding: 0.75rem 1rem;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #3f4147;
}}
body {{ background-color: #343541; color: white; margin-top: 80px; font-size: 18px; }}
.video-wrapper {{ display: flex; justify-content: center; padding: 1rem 0; }}
.video-wrapper video {{ opacity: 0; animation: fadeIn 1s ease-in-out forwards; }}
@keyframes fadeIn {{ to {{ opacity: 1; }} }}
</style>
""", unsafe_allow_html=True)

# Video container
video_container = st.empty()
if st.session_state["video_url"]:
    video_container.markdown(f'''
        <div class="video-wrapper">
            <video controls autoplay muted playsinline width="512">
                <source src="{st.session_state["video_url"]}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    ''', unsafe_allow_html=True)
else:
    video_container.markdown('<div class="video-wrapper"><em>Waiting for your first question...</em></div>', unsafe_allow_html=True)

# Generate D-ID video
def generate_did_video(text):
    headers = {
        "Authorization": f"Bearer {DID_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "script": {
            "type": "text",
            "input": text,
            "provider": {"type": "elevenlabs", "voice_id": "Rachel"}
        },
        "source_url": "https://create-images-results.d-id.com/DefaultPresenter.png"
    }
    try:
        response = requests.post("https://api.d-id.com/talks", json=payload, headers=headers)
        talk_id = response.json().get("id")
        for _ in range(20):
            check = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
            data = check.json()
            if data.get("result_url"):
                return data["result_url"]
            time.sleep(1)
    except Exception as e:
        print(f"[D-ID ERROR]: {e}")
    return None

# Prompt formatting
def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach...
QUERY: {query}
ANSWER: {base_answer}
... [trimmed for brevity]
"""
    return llm.invoke(prompt).content

# LangChain QA setup
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant that answers questions strictly based on the provided context.
If the context does not contain sufficient information to answer, respond with:
"I do not have sufficient information on this topic in the current knowledge base. Please consult HR or an expert for guidance."

Context:
{context}

Question:
{question}

Answer:
"""
)

try:
    retriever = load_faiss_index().as_retriever(return_source_documents=True)

    chain_type_kwargs = {
        "prompt": qa_prompt,
        "document_variable_name": "context"
    }

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

except Exception as e:
    st.error(f"Error loading FAISS index or QA chain: {e}")
    st.stop()


# Question input
with st.form("query_form", clear_on_submit=True):
    query = st.text_input("Ask your question:", placeholder="Type your question and click 'Submit'")
    submitted = st.form_submit_button("Submit")

if submitted:
    cleaned_query = query.strip()
    if cleaned_query:
        with st.spinner("Retrieving and formatting response..."):
            result = qa_chain({"query": cleaned_query})
            formatted = format_response(result["result"], cleaned_query)
            with st.spinner("Generating video..."):
                st.session_state["video_url"] = generate_did_video(formatted)
                st.session_state["latest_question"] = cleaned_query
                st.rerun()
    else:
        st.warning("Please enter a valid question.")

# Starter questions
with st.expander("Sample questions to get you started"):
    for i, q in enumerate([
        "How do I motivate an anxious type?",
        "How do I motivate an avoidant type?",
        "How to give feedback to an employee who dodges accountability?",
        "How do I deliver bad news without shutting someone down?"
    ]):
        if st.button(q, key=f"example_{i}"):
            st.session_state.latest_question = q

# Logout
if st.button("Logout"):
    for key in ["authenticated", "email", "video_url", "latest_question"]:
        st.session_state.pop(key, None)
    st.rerun()

# Footer logo
st.markdown("""
<div style="width: 100%; text-align: center; margin-top: 2rem;">
  <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png"
       style="max-width: 20%; height: auto;">
</div>
""", unsafe_allow_html=True)
