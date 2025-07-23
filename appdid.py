import streamlit as st
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag_pipeline import load_faiss_index
from did_utils import generate_did_video

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")

# Init LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Format LLM response
def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach with deep expertise in attachment theory and interpersonal motivation.

Imagine the following middle manager, who has little knowledge of psychology or communication theory, has asked you a question:

QUERY: {query}
ANSWER: {base_answer}

Respond as if you're speaking directly to the manager in a conversational, coaching style.

Base your suggestions strictly on the retrieved information provided in the ANSWER above. If the retrieved information does not sufficiently address the question, indicate that the current knowledge base does not cover this and suggest consulting HR or an expert.

1. Begin with a short, calm, and encouraging summary of your interpretation of their situation and your high-level advice — spoken in the voice of a soft-spoken, respected military officer who inspires loyalty and confidence.
2. Do not use academic terms, psychological jargon, or labels.
3. Provide 4 example phrases the manager could say, formatted as:

[Number]. "[Phrase]" – [Short reason why it works]

4. End with a reflective coaching question.

Respond in Markdown with calm, warm, concrete, dignified tone.
"""
    return llm.invoke(prompt).content

# User auth settings
UNRESTRICTED_EMAIL = "duffwarrenconsulting@gmail.com"
REQUIRED_PASSWORD = "b@6J8KJNff9*&^N:ll3Fb@r@3"
ACCESS_DURATION_DAYS = 7
TIMEOUT_SECONDS = 600
DB_PATH = "user_access.json"

# Load access DB
db = {}
try:
    with open(DB_PATH, "r") as f:
        db = json.load(f)
except FileNotFoundError:
    pass

# Init session state
for k, v in {"authenticated": False, "email": "", "history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🎥 Employee Management Video Assistant")

# Authentication
def authenticate_user(email):
    user_record = db.get(email)

    # Handle legacy string format
    if isinstance(user_record, str):
        expiry_str = user_record
        last_activity_str = datetime.now().isoformat()
    elif isinstance(user_record, dict):
        expiry_str = user_record.get("expiry")
        last_activity_str = user_record.get("last_activity", datetime.now().isoformat())
    else:
        expiry_str = (datetime.now() + timedelta(days=ACCESS_DURATION_DAYS)).isoformat()
        last_activity_str = datetime.now().isoformat()
        user_record = {"expiry": expiry_str, "last_activity": last_activity_str}
        db[email] = user_record
        with open(DB_PATH, "w") as f:
            json.dump(db, f)

    try:
        expiry_dt = datetime.fromisoformat(expiry_str)
        last_activity_dt = datetime.fromisoformat(last_activity_str)
    except Exception:
        st.error("Failed to parse user access record.")
        st.stop()

    if datetime.now() > expiry_dt:
        st.error("❌ Trial expired. Contact us to extend access.")
        st.stop()
    elif (datetime.now() - last_activity_dt).total_seconds() > TIMEOUT_SECONDS:
        st.warning("🔒 Session expired due to inactivity. Please log in again.")
        st.session_state.authenticated = False
        st.stop()
    else:
        user_record["last_activity"] = datetime.now().isoformat()
        db[email] = user_record
        with open(DB_PATH, "w") as f:
            json.dump(db, f)
        st.success(f"Access granted until {expiry_dt.date()}")
        st.session_state.authenticated = True
        st.experimental_rerun()

if not st.session_state.authenticated:
    st.subheader("Login for access")
    email = st.text_input("Email:")
    if email:
        st.session_state.email = email
        if email == UNRESTRICTED_EMAIL:
            with st.form("admin_login_form"):
                pwd = st.text_input("Admin password:", type="password")
                submitted = st.form_submit_button("Login")
            if submitted:
                if pwd == REQUIRED_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect admin password.")
        else:
            authenticate_user(email)

if not st.session_state.authenticated:
    st.stop()

# Load retriever
try:
    retriever = load_faiss_index().as_retriever(return_source_documents=True)
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an assistant that answers questions strictly based on the provided context.
If the context is insufficient, respond with:
"I do not have sufficient information on this topic in the current knowledge base. Please consult HR or an expert."

Context:
{context}

Question:
{question}

Answer:
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )
except Exception as e:
    st.error(f"RAG loading failed: {e}")
    st.stop()

# 🎬 Intro Video
if "intro_video_url" not in st.session_state:
    intro_script = """
Hello. I’m your Employee Management Assistant.

This space is for asking practical questions about how to better motivate and understand your team — especially when dealing with different personalities or emotional responses. 

You can ask things like:
- “How do I give tough feedback without triggering anxiety?”
- “How can I motivate someone who seems emotionally distant?”
- “What should I say when someone shuts down in a meeting?”

Let’s get started.
"""
    with st.spinner("Generating intro video..."):
        try:
            st.session_state.intro_video_url = generate_did_video(intro_script)
        except Exception:
            st.warning("Video intro generation failed.")
            st.session_state.intro_video_url = None

if st.session_state.intro_video_url:
    st.video(st.session_state.intro_video_url)

# 📝 User Interaction
with st.form("query_form", clear_on_submit=True):
    query = st.text_input("Ask your question here:")
    submitted = st.form_submit_button("Submit")

if submitted and query.strip():
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        formatted = format_response(result["result"], query)

        try:
            video_url = generate_did_video(formatted)
        except Exception:
            video_url = None
            st.warning("Video generation failed. Showing text only.")

        st.session_state.history.append({
            "q": query,
            "a": formatted,
            "video": video_url
        })
        st.experimental_rerun()

if st.session_state.history:
    st.markdown("### 📟 Previous Questions")
    for h in reversed(st.session_state.history):
        st.markdown(f"**🔨 Question:** {h['q']}")
        st.markdown(h["a"])
        if h.get("video"):
            st.video(h["video"])
        st.markdown("---")

if st.button("Clear History"):
    st.session_state.history = []
