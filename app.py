# app.py
import os
import json
from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from rag_pipeline import load_faiss_index

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Employee Management Assistant", layout="centered")

# ------------------------------------------------------------------------------
# Session state init (BEFORE any access)
# ------------------------------------------------------------------------------
for key, default in {
    "authenticated": False,
    "email": "",
    "submitted_query": "",
    "history": [],
    "latest_question": "",
    "speak_text": "",  # <- what the Agent will say on each render
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------------------
# Secrets / environment helpers
# ------------------------------------------------------------------------------
load_dotenv()

def _get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY    = _get_secret("OPENAI_API_KEY", "")
REQUIRED_PASSWORD = _get_secret("REQUIRED_PASSWORD", "")
DID_AGENT_ID      = _get_secret("DID_AGENT_ID", "")
DID_CLIENT_KEY    = _get_secret("DID_CLIENT_KEY", "")

# ------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# ------------------------------------------------------------------------------
# Global CSS
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
    display: flex; flex-direction: column; align-items: center; padding: 1rem 0;
}
#agent-video {
    width: 100%;
    max-width: 640px;      /* larger player */
    height: auto;
    min-height: 360px;     /* reserve space so it never collapses */
    object-fit: contain;   /* avoid cropping */
    border-radius: 12px;
    background: #000;
    opacity: 0; animation: fadeIn 0.6s ease-in-out forwards;
}
@keyframes fadeIn { to { opacity: 1; } }
.unmute-tip {
    text-align: right;
    width: 100%;
    max-width: 640px;
    margin: .25rem auto 0;
}
.unmute-btn {
    display: inline-block;
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #2b2d31;
    color: #ddd;
    border: 1px solid #3f4147;
    cursor: pointer;
    user-select: none;
}
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
# Authentication & trial access
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

# Active session checks
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
# Prompt & retriever
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
# D-ID Agents SDK embed (single video at top)
#   - On each rerender, if st.session_state["speak_text"] has content,
#     the agent connects and speaks that text.
# ------------------------------------------------------------------------------
from json import dumps as json_dumps

speak_text = st.session_state.get("speak_text", "")
escaped_text = json_dumps(speak_text)

agent_html = f"""
<style>
  .video-wrapper {{
    display:flex; flex-direction:column; align-items:center; gap:.5rem; padding:.75rem 0;
  }}
  #agent-video {{
    width:100%; max-width:640px; aspect-ratio:16/9; background:#000; border-radius:12px;
  }}
  .controls-row {{ display:flex; gap:.5rem; align-items:center; max-width:640px; width:100%; flex-wrap:wrap; }}
  .pill {{ font-size:.85rem; padding:.35rem .6rem; border-radius:999px; border:1px solid #3f4147; background:#2b2d31; color:#ddd; }}
  .btn  {{ cursor:pointer; user-select:none; }}
  .status-ok {{ color:#9be69b; }}
  .status-err {{ color:#ff9aa2; }}
  code.min {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; }}
</style>

<div class="video-wrapper">
  <video id="agent-video" playsinline muted></video>

  <div class="controls-row">
    <span id="status" class="pill">Status: <em>idle</em></span>
    <span id="error" class="pill status-err" style="display:none"></span>
  </div>

  <div class="controls-row">
    <span id="connect" class="pill btn">🔌 Connect</span>
    <span id="unmute"  class="pill btn" style="display:none">🔊 Unmute</span>
    <span id="speak"   class="pill btn">▶️ Speak response</span>
  </div>

  <div class="controls-row" style="opacity:.7;">Text to speak:&nbsp;
    <code class="min">{escaped_text}</code>
  </div>
</div>

<script type="module">
  import * as sdk from "https://cdn.jsdelivr.net/npm/@d-id/client-sdk@latest/dist/index.min.js";

  const videoEl   = document.getElementById("agent-video");
  const statusEl  = document.getElementById("status");
  const errorEl   = document.getElementById("error");
  const unmuteEl  = document.getElementById("unmute");
  const speakEl   = document.getElementById("speak");
  const connectEl = document.getElementById("connect");

  let agentManager = null;
  let srcObjectRef = null;
  let isConnected  = false;
  const speakText  = {escaped_text};

  function setStatus(text, ok=true) {{
    statusEl.innerHTML = "Status: " + text;
    statusEl.classList.toggle("status-ok", ok);
    statusEl.classList.toggle("status-err", !ok);
  }}
  function showError(msg) {{
    if (!msg) return;
    errorEl.style.display = "inline-block";
    errorEl.textContent = msg;
  }}
  function clearError() {{
    errorEl.style.display = "none";
    errorEl.textContent = "";
  }}

  async function tryUnmute() {{
    try {{
      videoEl.muted = false;
      await videoEl.play();
      unmuteEl.style.display = "none";
    }} catch (e) {{
      // Autoplay with sound blocked: keep muted, show button
      videoEl.muted = true;
      unmuteEl.style.display = "inline-block";
    }}
  }}
  unmuteEl.addEventListener("click", tryUnmute);

  speakEl.addEventListener("click", async () => {{
    clearError();
    if (!isConnected || !agentManager) {{
      showError("Please connect to the agent first");
      return;
    }}
    if (!speakText || speakText.length === 0) {{
      showError("No text to speak");
      return;
    }}
    setStatus("speaking…");
    try {{
      await agentManager.speak({{ type:"text", input:speakText }});
      setStatus("connected");
    }} catch (e) {{
      console.error("speak() error:", e);
      showError(e?.description || e?.message || "Speak failed");
      setStatus("error", false);
    }}
  }});

  const callbacks = {{
    onSrcObjectReady(value) {{
      srcObjectRef = value;
      videoEl.srcObject = value;
      return value;
    }},
    onVideoStateChange(state) {{
      console.log("onVideoStateChange:", state);
      if (state === "STOP") {{
        videoEl.srcObject = null;
        if (agentManager?.agent?.presenter?.idle_video) {{
          videoEl.src = agentManager.agent.presenter.idle_video;
        }}
      }} else {{
        videoEl.src = "";
        videoEl.srcObject = srcObjectRef;
      }}
    }},
    onConnectionStateChange(state) {{
      console.log("onConnectionStateChange:", state);
      if (state === "connected") {{
        isConnected = true;
        setStatus("connected");
        tryUnmute();
        // Auto-speak once on connect if we have text
        if (speakText && !window.__didAutoSpeak) {{
          window.__didAutoSpeak = true;
          speakEl.click();
        }}
      }} else {{
        setStatus(state, state !== "error");
      }}
    }},
    onError(error, data) {{
      console.error("D-ID SDK Error:", error, data);
      showError(error?.description || error?.message || "SDK error");
      setStatus("error", false);
    }},
  }};

  const auth = {{ type:"key", clientKey:"{DID_CLIENT_KEY}" }};
  const streamOptions = {{ compatibilityMode:"auto", streamWarmup:false }};

  async function connectOnce() {{
    clearError();
    setStatus("connecting…");
    try {{
      if (!agentManager) {{
        agentManager = await sdk.createAgentManager("{DID_AGENT_ID}", {{ auth, callbacks, streamOptions }});
      }}
      await agentManager.connect();   // If a session was stale, SDK will create a new one
    }} catch (e) {{
      console.error("connect() error:", e);
      showError(e?.description || e?.message || "Connect failed");
      setStatus("error", false);
    }}
  }}

  // Let the user explicitly start the connection and also retry on demand
  connectEl.addEventListener("click", connectOnce);

  // Optional: try once automatically on load, but the user can always retry
  connectOnce();
</script>
"""

# Render the agent video block (single window at the top)
components.html(agent_html, height=560)

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

        st.session_state["latest_question"] = cleaned_query
        st.session_state["speak_text"] = formatted
        st.rerun()
    else:
        st.warning("Please enter a valid question.")

# ------------------------------------------------------------------------------
# Sample questions (optional)
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
    for key in ["authenticated", "email", "latest_question", "speak_text"]:
        st.session_state.pop(key, None)
    st.rerun()

st.markdown("""
<div style="width: 100%; text-align: center; margin-top: 2rem;">
  <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png"
       style="max-width: 20%; height: auto;">
</div>
""", unsafe_allow_html=True)
