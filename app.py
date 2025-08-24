# ------------------------------------------------------------------------------
# app.py
# ------------------------------------------------------------------------------
import os
import json
from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import re   
import random  

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from rag_pipeline import load_faiss_index

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Employee Management Assistant", layout="centered")
# Shared content width for desktop/tablet
CONTENT_WIDTH = 768
VIDEO_AR = 16 / 9
VIDEO_HEIGHT = int(CONTENT_WIDTH / VIDEO_AR)  # ~405 or ~432
CONTROL_ROW_PX = 64
COMPONENT_VPAD_PX = 16
STRICT_INDEX_ONLY = True  # If True, never use model knowledge beyond FAISS context

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
    "chat_history": [],  # <- stores prior questions and answers for follow-up
    "awaiting_demographics": False,  # whether last turn asked a clarifying demo question
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
st.markdown(f"""
<style>
  :root {{ --content-width: {CONTENT_WIDTH}px; }}

  /* Make the page container the same width as the form/video on desktop */
  [data-testid="stAppViewContainer"] .main .block-container {{
    max-width: var(--content-width) !important;
    margin: 0 auto !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }}

  /* Make *all* custom component iframes fill the container width.
     Cover multiple DOM patterns Streamlit uses across versions. */
  [data-testid="stIFrame"] > iframe,
  [data-testid="stIFrame"] iframe,
  iframe[data-testid="stIFrame"],
  iframe[title^="st.iframe"] {{
    width: 100% !important;
    max-width: 100% !important;
    display: block !important;
  }}

  /* Mobile: go full width */
  @media (max-width: 768px) {{
    [data-testid="stAppViewContainer"] .main .block-container {{
      max-width: 100% !important;
      padding-left: 0 !important;
      padding-right: 0 !important;
    }}
  }}
</style>
""", unsafe_allow_html=True)

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
You answer ONLY from the provided context. If the answer is not in the
context, reply exactly: "THIS IS QUESTION IS OUTSIDE OF MY TRAINING". Do not add outside knowledge.

Context:
{context}

Question:
{question}

Answer (from context or INSUFFICIENT):
"""
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
# Define the sanitizer (near your other helpers)
# ------------------------------------------------------------------------------
def to_tts(text: str) -> str:
    """Strip Markdown-ish symbols and extra whitespace for clean TTS."""
    text = re.sub(r"[#*_`>•\-]+", " ", text)  # remove bullets/markdown chars
    text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
    return text

# Clinical terms function
def detect_clinical_terms(text: str) -> bool:
    clinical_keywords = [
        "attachment", "avoidant", "anxious", "disorganized",
        "cortisol", "dopamine", "oxytocin", "secure base", "trauma", "hypervigilant"
    ]
    lowered = text.lower()
    return any(term in lowered for term in clinical_keywords)

# ------------------------------------------------------------------------------
# Add closing variations helper function
# ------------------------------------------------------------------------------

def closing_variation() -> str:
    options = [
        "Try these and let me know how it works out. We can dial this in over time.",
        "Test these out and circle back. We’ll keep refining how you connect with this person.",
        "Put a few of these into play and observe what shifts. We’ll tune it further as needed.",
        "Use these as a starting point. I’m here to help you adjust as things evolve.",
        "Give these a try and see what happens. We'll keep calibrating as you go.",
        "Try them out and pay attention to what works. This is a process we can refine together.",
        "Experiment with these and bring your results. We’ll adapt based on what you notice.",
    ]
    return random.choice(options)

# ----------------------------------------------------------------------
# Enhanced response formatting: coaching + fallback LLM synthesis
# ----------------------------------------------------------------------
def format_response(base_answer: str, query: str, use_clinical: bool = False) -> str:
    """
    Returns either:
      - A single clarifying question (when demographics are missing), or
      - Tailored advice text that includes example phrases.
    Uses tags [CLARIFY] / [ADVICE] to decide whether to append the closing variation.
    Also injects known demographics and recent dialogue from session memory.
    """
    fallback_trigger = "THIS IS QUESTION IS OUTSIDE OF MY TRAINING"

    # ---- pull conversation memory (no changes to callers needed) ----
    _ensure_demo_state()
    demo = st.session_state.get("demographics", {})
    demo_str = ", ".join(
        s for s in [demo.get("age_or_gen"), demo.get("gender")] if s
    ) or "not provided"
    recent_dialogue = get_recent_dialogue(max_turns=6)

    def _postprocess(text: str):
        t = (text or "").strip()
        tag = None
        if t.startswith("[CLARIFY]"):
            tag = "CLARIFY"; t = t[len("[CLARIFY]"):].strip()
        elif t.startswith("[ADVICE]"):
            tag = "ADVICE";  t = t[len("[ADVICE]"):].strip()
        return t, tag

    # -------------------------
    # Fallback: no coverage in RAG
    # -------------------------
    if fallback_trigger in base_answer:
        if STRICT_INDEX_ONLY:
            return ("I don't have enough information in my knowledge base to answer that. "
                    "Try rephrasing or ask about a topic that's in the corpus.")

        prompt = f"""
You are an expert workplace communication coach trained in psychology and management.
The user is a busy middle manager asking about employee behavior and motivation.

KNOWN DEMOGRAPHICS (optional): {demo_str}
RECENT TURNS (for continuity; do not restate verbatim):
{recent_dialogue or '[none]'}

GOAL: Deliver a concise, actionable response. If the question is missing key
demographics (approximate age range / generation and gender, both OPTIONAL),
ask exactly ONE short clarifying question to tailor guidance, then STOP.
Do not provide advice until you get the answer to that question.

Demographics policy:
- Ask once, briefly, and make it clearly optional. Example:
  "Quick check: what's their approx. age range (e.g., 20s/Gen Z) and, if relevant, their gender?
   If you'd rather not say, I can proceed generally."
- When provided, use demographics to shape tone, channels, and motivators.
- Never stereotype; tie guidance to observable behaviors.

TONE: respectful, empathetic, professional.
LENGTH LIMITS:
- Clarifying question: Max 120 words (one sentence).
- Advice (when demographics are present): Max 200 words.

OUTPUT MODE (IMPORTANT):
- If demographics are MISSING in QUESTION: start with [CLARIFY] and output ONLY the clarifying question (one sentence).
- If demographics are PRESENT in QUESTION: start with [ADVICE] and output tailored advice that includes:
  - "Here are some example phrases." with 3 items, each followed by a one-sentence "why this works."

QUESTION:
{query}

Response:
"""
        response = llm.invoke(prompt).content.strip()
        cleaned, tag = _postprocess(response)
        # Maintain a simple flag so the next user message is treated as demographics
        st.session_state["awaiting_demographics"] = (tag == "CLARIFY")
        if tag == "CLARIFY":
            return cleaned
        return f"{cleaned} {closing_variation()}"

    # -------------------------
    # Main coaching response (RAG context present)
    # -------------------------
    clinical_note = """
Use accurate psychological terms (e.g., 'disorganized attachment') and refer to hormonal
factors when relevant (e.g., cortisol, oxytocin, etc.).
""" if use_clinical else """
Use practical, accessible language with no labels or jargon. Speak like a seasoned team leader.
"""

    prompt = f"""
You are a management communication coach. Your job is to guide a middle manager in handling
the issue below. You will speak using ONLY the knowledge provided in ANSWER.

KNOWN DEMOGRAPHICS (optional): {demo_str}
RECENT TURNS (for continuity; do not restate verbatim):
{recent_dialogue or '[none]'}

AUDIENCE: busy middle manager with no psychology background.
VOICE: calm, confident, supportive.
STYLE: short sentences, no jargon unless asked for it.

{clinical_note}

Demographics policy:
- Check QUESTION for approximate age range/generation and gender (OPTIONAL).
- If both are missing, ask exactly ONE short clarifying question, then STOP. Do not provide advice yet.
  Suggested pattern: "Quick check: what's their approx. age range (e.g., 20s/Gen Z) and, if relevant, their gender?
  If you'd rather not say, I can proceed generally."
- If demographics are present (or the manager declines), proceed with tailored advice.
- Use demographics only to adjust tone, channels, and likely motivators; avoid stereotypes. Base the "why" on behaviors.

FORMAT (when giving advice):
- Start with a brief explanation of what might be happening.
- Say: "Here are some example phrases." Then list 3 things the manager could say.
  After each, add "why this works" that refers to behavior patterns (and demographic cues if provided).

LENGTH LIMITS:
- Clarifying question: Max 120 words (one sentence).
- Advice: Max 200 words.

OUTPUT MODE (IMPORTANT):
- If demographics are MISSING in QUESTION: start with [CLARIFY] and output ONLY the clarifying question (one sentence).
- If demographics are PRESENT or user declines: start with [ADVICE] and output the advice in the specified format.

QUESTION: {query}

ANSWER (retrieved knowledge base):
{base_answer}

Response:
"""
    response = llm.invoke(prompt).content.strip()
    cleaned, tag = _postprocess(response)
    # Maintain a simple flag so the next user message is treated as demographics
    st.session_state["awaiting_demographics"] = (tag == "CLARIFY")
    if tag == "CLARIFY":
        return cleaned
    return f"{cleaned} {closing_variation()}"

# follow-up logic function

def generate_followups(history: list, current_q: str, current_a: str) -> list:
    """Return 2–3 follow-up coaching questions based on history and current Q/A."""
    prompt = f"""
You are a coaching assistant helping managers explore workplace communication styles.
Given the user's latest question and your response, suggest 2–3 helpful follow-up questions.
Avoid repeating the same idea. Make each follow-up different in tone or angle.

User's question: {current_q}
Assistant's answer: {current_a}

Follow-up questions:
- 
"""
    followup_response = llm.invoke(prompt).content.strip()
    return [line.lstrip("- ").strip() for line in followup_response.splitlines() if line.startswith("- ")]

def context_is_relevant(source_docs, question) -> bool:
    """Return True if retrieved docs clearly help answer the question."""
    if not source_docs:
        return False
    # Keep short to avoid extra tokens; judge top ~3 docs
    joined = "\n\n".join(getattr(d, "page_content", "")[:500] for d in source_docs[:3])
    judge_prompt = f"""
You are a strict judge. Decide if the CONTEXT contains information that directly helps
answer the QUESTION. Output only one word: RELEVANT or IRRELEVANT.

QUESTION:
{question}

CONTEXT:
{joined}

Answer:
"""
    try:
        verdict = (llm.invoke(judge_prompt).content or "").strip().upper()
    except Exception:
        return False
    return verdict.startswith("RELEVANT")

# ------------------------------
# Conversation memory helpers
# ------------------------------
_DEMO_GEN_WORDS = r"(gen\s*[zalpha]|gen\s*z|zoomers?|millennials?|gen\s*y|gen\s*x|boomers?)"
_DEMO_AGE_RANGE = r"(\b(?:late|early|mid)\s*\d0s\b|\b\d{2}s\b|\bunder\s*25\b|\bover\s*50\b|\b[12]\d(?:\s*-\s*[12]\d)?\b)"
_DEMO_GENDER    = r"\b(female|woman|women|male|man|men|nonbinary|non-binary|nb|they/them|she/her|he/him|she|her|he|him)\b"

def _ensure_demo_state():
    if "demographics" not in st.session_state:
        st.session_state["demographics"] = {"age_or_gen": None, "gender": None}

def parse_demographics(text: str) -> dict:
    """Pull coarse age/generation and gender cues from free text."""
    import re
    t = (text or "").lower()
    age_or_gen = None
    gender = None

    m1 = re.search(_DEMO_GEN_WORDS, t, flags=re.I)
    m2 = re.search(_DEMO_AGE_RANGE, t, flags=re.I)
    if m1 or m2:
        age_or_gen = (m1.group(0) if m1 else m2.group(0)).strip()

    m3 = re.search(_DEMO_GENDER, t, flags=re.I)
    if m3:
        gender = m3.group(0).strip()

    return {"age_or_gen": age_or_gen, "gender": gender}

def update_demographics_from_text(text: str):
    """Update session memory with any demographics we can detect in this turn."""
    _ensure_demo_state()
    found = parse_demographics(text)
    if found.get("age_or_gen"):
        st.session_state["demographics"]["age_or_gen"] = found["age_or_gen"]
    if found.get("gender"):
        st.session_state["demographics"]["gender"] = found["gender"]

def get_recent_dialogue(max_turns: int = 6) -> str:
    """Return recent Q/A pairs as a compact transcript for prompt grounding."""
    hist = st.session_state.get("chat_history", [])
    if not hist:
        return ""
    # last N-1 pairs, exclude current unfinished turn
    pairs = hist[-max_turns:]
    lines = []
    for q, a in pairs:
        lines.append(f"User: {q}")
        # Keep it short to save tokens for TTS
        lines.append(f"Coach: {a[:300]}")
    return "\n".join(lines)

# ------------------------------------------------------------------------------
# D-ID Agents SDK embed (single video at top)
#   - On each rerender, if st.session_state["speak_text"] has content,
#     the agent connects and speaks that text.
# ------------------------------------------------------------------------------
from json import dumps as json_dumps
speak_text = st.session_state.get("speak_text", "")
escaped_text = json_dumps(speak_text)  # safe JSON for JS

# Force a fresh iframe whenever the speak_text changes
nonce = abs(hash(st.session_state.get("speak_text", ""))) % 1_000_000

agent_html = f"""
<div data-nonce="{nonce}" style="display:none;"></div>

<style>
  /* Make the iframe document and wrapper fill the height we pass from Streamlit */
  html, body {{
    height: 100%;
    margin: 0;
    padding: 0;
  }}

  .video-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;         /* center horizontally */
    justify-content: flex-start; /* video sits at the top, controls just under */
    width: 100%;
    height: 100%;                /* fill iframe height */
    padding: 0;
    box-sizing: border-box;
    gap: .5rem;
  }}

  /* Video gets all remaining height; controls get {CONTROL_ROW_PX}px */
  #agent-video {{
    display: block;
    margin: 0 auto;              /* center video */
    width: 100%;
    max-width: 100%;
    height: calc(100% - {CONTROL_ROW_PX}px);  /* <-- keeps room for controls */
    max-height: 100%;
    background: #000;
    border-radius: 12px;
    object-fit: contain;         /* letterbox, no cropping */
    opacity: 0;
    animation: fadeIn .6s ease forwards;
  }}

  .row {{
    display: flex;
    gap: .75rem;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    height: {CONTROL_ROW_PX}px;  /* <-- matches calc() above */
  }}

  .chip {{
    font-size: 12px;
    padding: .25rem .6rem;
    border-radius: 999px;
    border: 1px solid #3f4147;
    background: #2b2d31;
    color: #ddd;
  }}

  .slider-wrap {{
    display: flex;
    align-items: center;
    gap: .5rem;
  }}

  @keyframes fadeIn {{ to {{ opacity: 1; }} }}
</style>

<div class="video-wrapper">
  <video id="agent-video" muted autoplay playsinline></video>

  <div class="row">
    <span id="status" class="chip">Status: init</span>
    <div class="slider-wrap chip" style="background:#1f1f22;">
      <span>Vol</span>
      <input id="vol" type="range" min="0" max="1" step="0.05" value="0.8" style="accent-color:#6ea8fe;">
      <span id="muted" class="chip">muted</span>
      <span id="unmute-btn" class="chip" style="cursor:pointer;">🔊 Unmute</span>
    </div>
  </div>
</div>

<script type="module">
  import * as sdk from "https://cdn.jsdelivr.net/npm/@d-id/client-sdk@latest/dist/index.min.js";

  const videoEl   = document.getElementById("agent-video");
  const statusEl  = document.getElementById("status");
  const volEl     = document.getElementById("vol");
  const mutedEl   = document.getElementById("muted");
  const unmuteEl  = document.getElementById("unmute-btn");

  // speakText is injected from Streamlit (safe JSON string)
  const speakText = {escaped_text if 'escaped_text' in globals() else '" "'};

  let srcObjectRef = null;
  let agentManager = null;
  let connected    = false;

  const setStatus = (s) => statusEl.textContent = "Status: " + s;
  const updateAudioUI = () => {{
    mutedEl.textContent = videoEl.muted ? "muted" : "unmuted";
    unmuteEl.style.display = videoEl.muted ? "inline-block" : "none";
  }};

  const auth = {{ type: "key", clientKey: "{DID_CLIENT_KEY}" }};
  const streamOptions = {{ compatibilityMode: "auto", streamWarmup: false }};

  async function ensureConnected() {{
    if (connected) return;
    setStatus("connecting");
    agentManager = await sdk.createAgentManager("{DID_AGENT_ID}", {{
      auth,
      callbacks: {{
        onSrcObjectReady(value) {{
          srcObjectRef = value;
          videoEl.srcObject = value;
          videoEl.volume = parseFloat(volEl.value || "0.8");
          videoEl.muted = true;      // start muted to satisfy autoplay
          videoEl.play().catch(()=>{{}});
          updateAudioUI();
          return value;
        }},
        onConnectionStateChange(state) {{
          setStatus(state);
          connected = (state === "connected");
          if (connected && speakText && speakText.trim()) {{
            setTimeout(() => speakNow(speakText), 300);
          }}
        }},
        onError(error) {{
          setStatus("error");
          console.error("D-ID error:", error);
        }},
      }},
      streamOptions
    }});
    await agentManager.connect();
  }}

  async function speakNow(text) {{
    if (!connected) return;
    try {{
      if (srcObjectRef) {{
        videoEl.src = "";
        videoEl.srcObject = srcObjectRef;
        videoEl.play().catch(()=>{{}});
      }}
      await agentManager.speak({{ type: "text", input: text.slice(0, 900) }});
    }} catch (e) {{
      console.error("speak error:", e);
    }}
  }}

  async function tryUnmute() {{
    try {{
      videoEl.muted = false;
      await videoEl.play();
    }} catch (e) {{
      // If blocked by autoplay policy, remain muted until user clicks again
      videoEl.muted = true;
    }}
    updateAudioUI();
  }}

  // UI hooks
  unmuteEl.onclick = tryUnmute;
  videoEl.addEventListener("click", tryUnmute, {{ once: false }});
  volEl.oninput = () => {{
    videoEl.volume = parseFloat(volEl.value || "0.8");
  }};

  // Boot
  (async () => {{
    try {{
      await ensureConnected();
      await tryUnmute();   // best-effort; may still require a click
    }} catch (e) {{
      console.error("init error:", e);
      setStatus("failed to connect");
    }}
  }})();
</script>
"""

html_key = f"did_agent_{hash(st.session_state.get('speak_text', '')) % 1_000_000}"

components.html(
    agent_html,
    height=VIDEO_HEIGHT + CONTROL_ROW_PX + COMPONENT_VPAD_PX,
    scrolling=False
)

# ------------------------------------------------------------------------------
# Query input
# ------------------------------------------------------------------------------
with st.form("query_form", clear_on_submit=True):
    query = st.text_input("Ask your question:", placeholder="Type your question and click 'Submit'")
    clinical_mode_toggle = st.checkbox(
        "Use clinical terms (e.g., for coaching psychologists or HR specialists)", value=False
    )
    submitted = st.form_submit_button("Submit")

if submitted:
    cleaned_query = query.strip()
    if cleaned_query:
        update_demographics_from_text(cleaned_query)
        # If last turn asked for demographics, consider this turn the answer and clear the flag
        if st.session_state.get("awaiting_demographics"):
            st.session_state["awaiting_demographics"] = False

        with st.spinner("Retrieving and formatting response..."):
            try:
                result = qa_chain({"query": cleaned_query})
                base = (result.get("result") or result.get("answer") or "").strip()
            except Exception:
                st.error("Sorry, I hit an error fetching an answer. Playing a brief explanation instead.")
                base = ""
                result = {}

            # Guardrail: if retrieved docs aren’t actually relevant, force "insufficient"
            src_docs = result.get("source_documents") or []
            if not context_is_relevant(src_docs, cleaned_query):
                base = "THIS IS QUESTION IS OUTSIDE OF MY TRAINING"

            # If we didn't get anything from RAG, speak a short default line so the avatar plays
            if not base:
                spoken = "I’m here and ready. Please click Unmute if you can’t hear me."
            else:
                use_clinical = clinical_mode_toggle or detect_clinical_terms(cleaned_query)
                formatted = format_response(base, cleaned_query, use_clinical)

                # Learn any demographics the assistant echoed (e.g., "For your Gen Z analyst...")
                update_demographics_from_text(formatted)

                # --- TTS LENGTH-BUFFERED LOGIC ---
                MAX_TTS_LEN = 800
                if len(formatted) > MAX_TTS_LEN:
                    intro = formatted.split("Here are some example phrases.")[0].strip()
                    fallback = f"{intro} Here are some example phrases."
                    spoken = fallback[:MAX_TTS_LEN]
                else:
                    spoken = formatted[:MAX_TTS_LEN]

            # Ensure punctuation for better prosody
            if not spoken.strip().endswith(('.', '!', '?', '…')):
                spoken += "."

            # (Optional) write out for debugging
            with open("debug_tts.txt", "w", encoding="utf-8") as f:
                f.write(spoken)

            spoken = to_tts(spoken)
            st.session_state["speak_text"] = spoken

        # Keep history only when we had a real base answer (optional)
        if base:
            st.session_state["latest_question"] = cleaned_query
            st.session_state["chat_history"].append((cleaned_query, spoken))
            if len(st.session_state["chat_history"]) > 10:
                st.session_state["chat_history"] = st.session_state["chat_history"][-10:]
        st.rerun()
    else:
        st.warning("Please enter a valid question.")

# ------------------------------------------------------------------------------  
# Follow-up Suggestions 
# ------------------------------------------------------------------------------  
if "followups" in st.session_state and st.session_state["followups"]:  
    st.markdown("**Want to go deeper? Try one of these follow-up questions:**")  
    cols = st.columns(len(st.session_state["followups"]))  
    for i, followup in enumerate(st.session_state["followups"]):  
        with cols[i]:  
            if st.button(followup, key=f"followup_{i}"):  
                st.session_state["latest_question"] = followup
                
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
