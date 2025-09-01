import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Initialize HF client
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

st.set_page_config(page_title="Mistral Chatbot", page_icon="🤖")

st.title("🤖 Chat with Mistral-7B-Instruct")

# Store chat history in session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])

# Input box
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Query Mistral
    response = client.chat_completion(
        st.session_state["messages"],
        max_tokens=150,
        temperature=0.7
    )

    # Extract text
    reply = response.choices[0].message.content

    # Add assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
