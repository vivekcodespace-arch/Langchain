import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline('text-generation',model = 'facebook/opt-350m' ,device_map='cpu')

generator = load_model()

st.header("Simple chatbot")

user_input = st.text_input("Enter your query")

if user_input:
    response = generator(user_input,
                         max_new_tokens=100)
    st.write(response[0]['generated_text'])    