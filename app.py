import streamlit as st

st.set_page_config(page_title="PDF Q&A + ArXiv Agent", layout="wide")

import os
import json
from main import build_graph
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph app
app = build_graph()


st.sidebar.title("⚙️ Settings")

# Multi-user session
session_id = st.sidebar.text_input("👤 Session ID", value="default")
history_file = f"chat_history_{session_id}.json"

# Clear chat option
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.history = []
    if os.path.exists(history_file):
        os.remove(history_file)
    st.rerun()


def load_chat_history(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        raw = json.load(f)
    messages = []
    for m in raw:
        if m["type"] == "human":
            messages.append(HumanMessage(content=m["content"]))
        elif m["type"] == "ai":
            messages.append(AIMessage(content=m["content"]))
    return messages

def save_chat_history(messages, path):
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            formatted.append({"type": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            formatted.append({"type": "ai", "content": m.content})
    with open(path, "w") as f:
        json.dump(formatted, f, indent=2)

#streamlit chat ui
st.title("📄 PDF Q&A Agent with arXiv Search")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = load_chat_history(history_file)

# Display history
for m in st.session_state.history:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# Chat input
query = st.chat_input("Ask a question...")

if query:
    # Show user message
    st.chat_message("user").markdown(query)
    st.session_state.history.append(HumanMessage(content=query))

    # Get agent reply
    result = app.invoke({"messages": st.session_state.history})
    reply = result["messages"][-1].content

    # Show assistant reply
    st.chat_message("assistant").markdown(reply)
    st.session_state.history.append(AIMessage(content=reply))

    # Save chat
    save_chat_history(st.session_state.history, history_file)
