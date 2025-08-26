from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# Load TinyLlama locally
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    max_new_tokens=100
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Direct invoke (no ChatHuggingFace needed)
result = llm.invoke("What's the capital of India?")
print(result)
