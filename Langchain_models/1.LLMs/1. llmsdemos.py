from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()


# ✅ Use an available Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")   # Or "gemini-1.5-flash"

result = llm.invoke("What's the capital of India?")

print(result)
