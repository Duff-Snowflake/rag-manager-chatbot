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
from langchain.prompts import PromptTemplate

# Load environment and initialize LLM
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

def format_response(base_answer, query):
    prompt = f"""
You are a management communication coach with deep expertise in attachment theory and interpersonal motivation.

Imagine the following middle manager, who has little knowledge of psychology or communication theory, has asked you a question:

QUERY: {query}
ANSWER: {base_answer}

Respond as if you're speaking directly to the manager in a conversational, coaching style.

Base your suggestions strictly on the retrieved information provided in the ANSWER above. If the retrieved information does not sufficiently address the question, indicate that the current knowledge base does not cover this and suggest consulting HR or an expert. Do not invent answers or use general knowledge not contained in the retrieved material.

1. Begin with a short, calm, and encouraging summary of your interpretation of their situation and your high-level advice — spoken in the voice of a soft-spoken, respected military officer who inspires loyalty and confidence.
2. Do not use academic terms, psychological jargon, or labels. Avoid condescension or overexplaining.
3. Then, naturally introduce 4 specific example phrases the manager could say in this situation. For each phrase, format as:

[Number]. "[Example phrase]" – [Short explanation why it works, in plain language managers understand]

4. End with a short reflective question prompting the manager to consider how their tone and word choice can create safety and motivation in their employee.
5. Write the full response as markdown, in a voice that feels calm, warm, confident, and deeply respectful of the manager’s desire to motivate their team effectively.

Avoid repeating the query or answer unless it's helpful to reframe. Be clear, empathetic, concrete, and dignified.
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
body { background-color: #343541; color: white; margin-top: 80px; font-size: 18px; }
.top-banner {
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
}
.top-banner img {
    height: 30px;
    margin-right: 10px;
}
.top-banner .status {
    font-size: 0.85rem;
    color: #b3b3b3;
}
.chat-entry {
    margin-bottom: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.chat-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    background: #2f3136;
    border: 1px solid #565869;
}
.chat-question {
    align-self: flex-end;
    background-color: #40414f;
    color: #fff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-size: 18px; 
}
.chat-response {
    align-self: flex-start;
    background-color: #444654;
    color: #fff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 18px; 
}
.chat-response * {
    font-size: inherit !important;  /* force children to inherit the size */
}
</style>

<div class="top-banner">
  <div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png" alt="Logo">
    <strong>Employee Management Assistant</strong>
  </div>
  <div class="status">
    Response added. Logged in as: {email}
  </div>
</div>
""".replace("{email}", st.session_state.get("email","")), unsafe_allow_html=True)

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

# Define a strict QA prompt to enforce source-based answers only
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

# Initialize retriever and QA chain
try:
    retriever = load_faiss_index().as_retriever(return_source_documents=True)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True,chain_type_kwargs={"prompt": qa_prompt})
except Exception as e:
    st.error(f"Error loading FAISS index or setting up RetrievalQA: {e}")
    st.stop()

# Conversation area (oldest at top, newest at bottom)
if st.session_state.history:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for entry in st.session_state.history:
        st.markdown(f'''
            <div class="chat-entry">
                <div class="chat-question">{entry["q"]}</div>
                <div class="chat-response">{entry["a"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom interaction area (Query input form)
with st.form("query_form", clear_on_submit=True):
    query = st.text_input(
        "Ask your question:",
        placeholder="Type your question and click 'Submit'"
    )
    submitted = st.form_submit_button("Submit")

if submitted:
    cleaned_query = query.strip()
    if cleaned_query:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": cleaned_query})
            formatted = format_response(result["result"], cleaned_query)
            st.session_state.history.append({
                "q": cleaned_query,
                "a": formatted,
                "t": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sources": result.get("source_documents", [])
            })
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

# Logout button
if st.button("Logout", key="logout"):
    for key in ["authenticated", "email", "history", "submitted_query"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# Clear history button
if st.button("Clear History", key="clear_button"):
    st.session_state.history = []
    st.write("History cleared. Please refresh if not updated.")

# Bottom logo display
st.markdown("""
<div style="width: 100%; text-align: center; margin-top: 2rem;">
  <img src="https://raw.githubusercontent.com/Duff-Snowflake/rag-manager-chatbot/main/assets/Your_logo_here001.png"
       style="max-width: 20%; height: auto;">
</div>
""", unsafe_allow_html=True)