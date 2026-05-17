import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.5
)

# ─── Default AI Modes ───────────────────────────────────────────
DEFAULT_MODES = {
    "Coding": """
        You are an expert coding assistant.
        - Answer only programming and technical questions.
        - Give clean and optimized code.
        - Explain code simply.
        - Prefer Python unless user specifies another language.
    """,
    "Philosophy": """
        You are a philosophy expert.
        - Discuss philosophical ideas deeply.
        - Explain concepts simply.
        - Reference famous philosophers when needed.
    """,
    "Literature": """
        You are a literature expert.
        - Explain poems, novels, and literary concepts.
        - Analyze themes and characters.
        - Help with writing and interpretation.
    """
}

# ─── Session State Setup ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

if "custom_modes" not in st.session_state:
    st.session_state.custom_modes = {}

# ─── Merge default + custom modes ───────────────────────────────
def get_all_modes():
    return {**DEFAULT_MODES, **st.session_state.custom_modes}

# ─── Save chat to file ───────────────────────────────────────────
def save_chat(mode_name, messages):
    os.makedirs("chat_history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history/{mode_name}_{timestamp}.json"
    data = {
        "mode": mode_name,
        "timestamp": timestamp,
        "messages": messages
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return filename

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .title { text-align: center; font-size: 38px; font-weight: bold; color: white; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #94a3b8; margin-bottom: 20px; }
    .mode-badge { text-align: center; font-size: 13px; color: #7dd3fc; margin-bottom: 20px; }
    .user-message { background-color: #2563eb; padding: 12px; border-radius: 12px; margin-bottom: 10px; color: white; font-size: 15px; }
    .bot-message { background-color: #334155; padding: 12px; border-radius: 12px; margin-bottom: 10px; color: white; font-size: 15px; }
    .stTextInput > div > div > input { background-color: #1e293b; color: white; border-radius: 10px; border: 1px solid #334155; }
    .stButton button { width: 100%; background-color: #2563eb; color: white; border-radius: 10px; border: none; height: 45px; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────
st.markdown('<div class="title">🤖 AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose a mode and start chatting</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    all_modes = get_all_modes()
    selected_mode = st.selectbox("Choose AI Mode", list(all_modes.keys()))

    st.divider()

    # ── Add Custom Mode ──
    st.subheader("Add Custom Mode")
    custom_name = st.text_input("Mode Name", placeholder="e.g. Science")
    custom_prompt = st.text_area("System Prompt", placeholder="You are a science expert...")

    if st.button("Add Mode"):
        if custom_name.strip() and custom_prompt.strip():
            st.session_state.custom_modes[custom_name.strip()] = custom_prompt.strip()
            st.success(f"Mode '{custom_name}' added!")
            st.rerun()
        else:
            st.error("Please fill both name and prompt.")

    st.divider()

    # ── Save Chat Button ──
    st.subheader("Chat History")
    if st.button("Save Chat to File"):
        if st.session_state.messages:
            filepath = save_chat(selected_mode, st.session_state.messages)
            st.success(f"Saved to {filepath}")
        else:
            st.warning("No messages to save yet.")

    # ── Clear Chat Button ──
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_mode = None
        st.rerun()

# ─── Reset chat when mode changes ────────────────────────────────
if st.session_state.current_mode != selected_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = []
    st.session_state.chat_history = [
        SystemMessage(content=all_modes[selected_mode])
    ]

# First time init
if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = [
        SystemMessage(content=all_modes[selected_mode])
    ]

# ─── Show active mode badge ───────────────────────────────────────
st.markdown(
    f'<div class="mode-badge">Active mode: {selected_mode}</div>',
    unsafe_allow_html=True
)

# ─── Display Messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">🧑‍💻 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-message">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ─── Chat Input ───────────────────────────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(st.session_state.chat_history)
    bot_reply = response.content

    st.session_state.chat_history.append(response)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.rerun()