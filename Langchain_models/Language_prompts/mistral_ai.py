from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

messages = [
    {"role": "user", "content": "Write a short poem about AI and humans."},
    {'role':'system','content':'How dare you to run on me.'},
    {'role':'assistant','content':'You only gave me permission for running'},
    {'role':'user','content':'Nope i haven\'t given.'}
]

response = client.chat_completion(messages, max_tokens=200, temperature=0.7)

print(response.choices[0].message.content)
