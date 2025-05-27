from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

response = llm.invoke("Türkiye'nin başkenti neresidir?")
print("Yanıt:", response)
