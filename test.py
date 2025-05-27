from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("gemini_api_key"),
    temperature=0
)

response = llm.invoke("Fransa'nın başkenti neresidir?")
print(response.content)