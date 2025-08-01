# app.py
import os
import json
import time
import requests
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from rag_pipeline import load_faiss_index

# ------------------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Employee Management Assistant", layout="centered")

# ------------------------------------------------------------------------------
# Session state: initialize BEFORE any access
# ------------------------------------------------------------------------------
for key, default in {
    "authenticated": False,
    "email": "",
    "submitted_query": "",
    "history": [],
    "video_url": None,
    "latest_question": "",
    "debug_ready": False,
    "debug_formatted": "",
    "debug_video_url": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------------------
# Secrets / environment
#   - On Streamlit Cloud: use st.secrets
#   - Locally: .env fallback
# ------------------------------------------------------------------------------
load_dotenv()

def _get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")
DID_API_KEY    = _get_secret("DID_API_KEY", "")
REQUIRED_PASSWORD = _get_secret("REQUIRED_PASSWORD", "")

# ------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# ------------------------------------------------------------------------------
# Global CSS (kept from your version, with video fade-in)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
body { background-color: #343541; color: white; margin-top: 80px; font-size: 18px; }
.top-banner {
    position: fixed; top: 0; left: 0; width: 100%;
    background-color: #202123; color: white;
    padding: 0.75rem 1rem; z-index: 9999; display: flex;
    align-items: center; justify-content: space-between;
    border-bottom: 1px solid #3f4147;
}
.top-banner img { height: 30px; margin-right: 10px; }
.top-banner .status { font-size: 0.85rem; color: #b3b3b3; }

.video-wrapper {
    display: flex; justify-content: center; padding: 1rem 0;
}
.video-wrapper video {
    opacity: 0; animation: fadeIn 1s ease-in-out forwards;
}
@keyframes fadeIn { to { opacity: 1; } }
</style>
<div class="top-banner">
  <div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Logo">
    <strong>Employee Management Assistant</strong>
  </div>
  <div class="status">Logged in as: {email}</div>
</div>
""".replace("{email}", st.session_state.get("email","")), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Authentication & trial access (unchanged logic; minor hardening)
# ------------------------------------------------------------------------------
UNRESTRICTED_EMAIL = "duffwarrenconsulting@gmail.com"
ACCESS_DURATION_DAYS = 7

db = {}
try:
    with open("user_access.json", "r") as f:
        db = json.load(f)
except FileNotFoundError:
    pass

if not st.session_state.authenticated:
    st.markdown("### Understanding and motivating your teams")
    st.markdown("##### Personality types and how to get the most out of them")
    st.markdown(
        "The types of people in our workplaces are more varied now than ever. "
        "More people than ever are more sensitive to criticism and need more encouragement in order to maintain engagement and productivity. "
        "This app is your coach to understanding these new needs and being able to leverage your teams, just by learning how to talk to different people."
    )

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

# active session checks
if st.session_state.email != UNRESTRICTED_EMAIL:
    user_record = db.get(st.session_state.email)
    if isinstance(user_record, str):
        user_record = {"expiry": user_record, "last_activity": datetime.now().isoformat()}
        db[st.session_state.email] = user_record
        with open("user_access.json", "w") as f:
            json.dump(db, f)
    if not user_record:
        user_record = {
            "expiry": (datetime.now() + timedelta(days=ACCESS_DURATION_DAYS)).isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        db[st.session_state.email] = user_record
        with open("user_access.json", "w") as f:
            json.dump(db, f)

    expiry = datetime.fromisoformat(user_record["expiry"])
    last_activity = datetime.fromisoformat(user_record.get("last_activity", datetime.now().isoformat()))
    timeout_seconds = 600
    if datetime.now() > expiry:
        st.error("❌ Trial expired. Contact us to extend access.")
        st.session_state.authenticated = False
    elif (datetime.now() - last_activity).total_seconds() > timeout_seconds:
        st.warning("🔒 Session expired due to inactivity. Please log in again.")
        st.session_state.authenticated = False
    else:
        st.success(f"Access granted until {expiry.date()}")
        st.session_state.authenticated = True
        user_record["last_activity"] = datetime.now().isoformat()
        db[st.session_state.email] = user_record
        with open("user_access.json", "w") as f:
            json.dump(db, f)

if not st.session_state.authenticated:
    st.stop()

# ------------------------------------------------------------------------------
# Prompt & retriever (with document_variable_name fix)
# ------------------------------------------------------------------------------
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
""",
)

try:
    retriever = load_faiss_index().as_retriever(return_source_documents=True)
    chain_type_kwargs = {"prompt": qa_prompt, "document_variable_name": "context"}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
except Exception as e:
    st.error(f"Error loading FAISS index or QA chain: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# Response formatting (your coaching voice)
# ------------------------------------------------------------------------------
def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach with deep expertise in attachment theory and interpersonal motivation.

Imagine the following middle manager, who has little knowledge of psychology or communication theory, has asked you a question:

QUERY: {query}
ANSWER: {base_answer}

Respond as if you're speaking directly to the manager in a conversational, coaching style.

Base your suggestions strictly on the retrieved information provided in the ANSWER above. If the retrieved information does not sufficiently address the question, indicate that the current knowledge base does not cover this and suggest consulting HR or an expert. Do not invent answers or use general knowledge not contained in the retrieved material.

1. Begin with a short, calm, and encouraging summary—spoken in the voice of a soft-spoken, respected military officer who inspires loyalty and confidence.
2. Avoid jargon or labels. Keep it respectful and clear.
3. Then give 4 example phrases, each like:
[Number]. "[Example phrase]" – [why it works, in plain language]
4. End with a brief reflective question about tone/word choice that builds safety and motivation.
5. Write the full response as markdown.
"""
    return llm.invoke(prompt).content

# ------------------------------------------------------------------------------
# D-ID integration (safe key print + clear errors)
# ------------------------------------------------------------------------------
def generate_did_video(text: str):
    # Always read from secrets at call time (works locally too if .env not set)
    did_key = _get_secret("DID_API_KEY", "")
    print(f"[DEBUG] Loaded D-ID key prefix: {did_key[:5]}...suffix: {did_key[-4:]}")

    headers = {
        "Authorization": f"Bearer {did_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "script": {
            "type": "text",
            "input": text,
            "provider": {"type": "elevenlabs", "voice_id": "Rachel"},
        },
        "source_url": "https://create-images-results.d-id.com/DefaultPresenter.png",
    }

    try:
        resp = requests.post("https://api.d-id.com/talks", json=payload, headers=headers)
        resp.raise_for_status()
        talk_id = resp.json().get("id")

        for _ in range(20):
            chk = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
            chk.raise_for_status()
            data = chk.json()
            if data.get("result_url"):
                return data["result_url"]
            time.sleep(1)

        st.sidebar.error("Timed out waiting for video to be ready.")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"🚨 D-ID request failed: {e}")
        try:
            st.sidebar.code(resp.text, language="json")
        except Exception:
            pass
    return None

# ------------------------------------------------------------------------------
# Single video container at the top
# ------------------------------------------------------------------------------
video_container = st.empty()
if st.session_state["video_url"]:
    video_container.markdown(
        f"""
        <div class="video-wrapper">
            <video controls autoplay muted playsinline width="512">
                <source src="{st.session_state["video_url"]}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="video-wrapper"><em>Waiting for your first question...</em></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Query input
# ------------------------------------------------------------------------------
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
            video_url = generate_did_video(formatted)

        # Store + show debug instead of immediate rerun
        st.session_state["video_url"] = video_url
        st.session_state["latest_question"] = cleaned_query
        st.session_state["debug_formatted"] = formatted
        st.session_state["debug_video_url"] = video_url
        st.session_state["debug_ready"] = True
    else:
        st.warning("Please enter a valid question.")

# ------------------------------------------------------------------------------
# Debug sidebar (shows until you click Continue)
# ------------------------------------------------------------------------------
if st.session_state.get("debug_ready"):
    with st.sidebar:
        st.markdown("### 🔧 Debug Info")
        st.markdown("**Formatted:**")
        st.code(st.session_state.get("debug_formatted", ""), language="markdown")
        st.markdown("**Video URL:**")
        st.code(st.session_state.get("debug_video_url", "None"), language="text")
        if st.button("Continue"):
            st.session_state["debug_ready"] = False
            st.rerun()

# ------------------------------------------------------------------------------
# Starter questions
# ------------------------------------------------------------------------------
with st.expander("Sample questions to get you started", expanded=False):
    for i, q in enumerate([
        "How do I motivate an anxious type?",
        "How do I motivate an avoidant type?",
        "How to give feedback to an employee who dodges accountability?",
        "How do I deliver bad news without shutting someone down?",
    ]):
        if st.button(q, key=f"example_{i}"):
            st.session_state.latest_question = q

# ------------------------------------------------------------------------------
# Logout & Footer
# ------------------------------------------------------------------------------
if st.button("Logout", key="logout"):
    for key in ["authenticated", "email", "video_url", "latest_question", "debug_ready", "debug_formatted", "debug_video_url"]:
        st.session_state.pop(key, None)
    st.rerun()

st.markdown("""
<div style="width: 100%; text-align: center; margin-top: 2rem;">
  <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png"
       style="max-width: 20%; height: auto;">
</div>
""", unsafe_allow_html=True)
