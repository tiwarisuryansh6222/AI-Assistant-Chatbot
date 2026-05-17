import streamlit as st
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

# For PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.5
)

# ─── Default AI Modes ─────────────────────────────────────────────
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

# ─── Session State Setup ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

if "custom_modes" not in st.session_state:
    st.session_state.custom_modes = {}

# ─── Helper Functions ─────────────────────────────────────────────
def get_all_modes():
    return {**DEFAULT_MODES, **st.session_state.custom_modes}

def generate_json_bytes(mode_name, messages):
    """Creates JSON content as bytes — for download button"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "mode": mode_name,
        "exported_at": timestamp,
        "messages": messages
    }
    return json.dumps(data, indent=2).encode("utf-8")

def generate_pdf_bytes(mode_name, messages):
    """Creates PDF content as bytes — for download button"""
    import io
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=6
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=10,
        textColor="#888888",
        spaceAfter=20
    )
    user_style = ParagraphStyle(
        "User",
        parent=styles["Normal"],
        fontSize=11,
        textColor="#1a56db",
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold"
    )
    bot_style = ParagraphStyle(
        "Bot",
        parent=styles["Normal"],
        fontSize=11,
        textColor="#1f2937",
        spaceBefore=4,
        spaceAfter=10
    )

    story = []
    story.append(Paragraph(f"Chat Export — {mode_name} Mode", title_style))
    story.append(Paragraph(
        f"Exported on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        meta_style
    ))

    for msg in messages:
        if msg["role"] == "user":
            story.append(Paragraph("You:", user_style))
        else:
            story.append(Paragraph("AI Assistant:", user_style))

        # Clean text for PDF (replace special chars)
        clean_text = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(clean_text, bot_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Persona AI",
    page_icon="🤖",
    layout="centered"
)

# ─── CSS ──────────────────────────────────────────────────────────
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
    .thinking-box { background-color: #1e293b; border-left: 3px solid #2563eb; padding: 10px 14px; border-radius: 8px; color: #94a3b8; font-size: 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────
st.markdown('<div class="title">🤖 Persona AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose a mode and start chatting</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────
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
            st.error("Please fill both fields.")

    st.divider()

    # ── Download Buttons ──
    st.subheader("Download Chat")

    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # JSON Download
        json_bytes = generate_json_bytes(selected_mode, st.session_state.messages)
        st.download_button(
            label="Download as JSON",
            data=json_bytes,
            file_name=f"chat_{selected_mode}_{timestamp}.json",
            mime="application/json"
        )

        # PDF Download
        pdf_bytes = generate_pdf_bytes(selected_mode, st.session_state.messages)
        st.download_button(
            label="Download as PDF",
            data=pdf_bytes,
            file_name=f"chat_{selected_mode}_{timestamp}.pdf",
            mime="application/pdf"
        )

    else:
        st.caption("No messages yet to download.")

    st.divider()

    # ── Clear Chat ──
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_mode = None
        st.rerun()

# ─── Reset on mode change ─────────────────────────────────────────
if st.session_state.current_mode != selected_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = []
    st.session_state.chat_history = [
        SystemMessage(content=all_modes[selected_mode])
    ]

if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = [
        SystemMessage(content=all_modes[selected_mode])
    ]

# ─── Active Mode Badge ────────────────────────────────────────────
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

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    st.markdown(
        f'<div class="user-message">🧑‍💻 {user_input}</div>',
        unsafe_allow_html=True
    )

    # Show thinking indicator while waiting
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(
        '<div class="thinking-box">🤖 AI is thinking...</div>',
        unsafe_allow_html=True
    )

    # Get AI response
    response = model.invoke(st.session_state.chat_history)
    bot_reply = response.content

    # Remove thinking indicator
    thinking_placeholder.empty()

    st.session_state.chat_history.append(response)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.rerun()