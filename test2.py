from base_processor import BaseProcessor
from dotenv import load_dotenv
import os

load_dotenv()

processor = BaseProcessor(
    index_root="faiss_indexes",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    google_config={
        "model": "gemini-2.0-flash",
        "gemini_api_key": os.getenv("GEMINI_API_KEY")
    }
)

response = processor.answer_question(
    question="AI Acts nedir?",
    category="AI_Acts",
    docs_path="data/AI_Acts"
)

print("Cevap:", response["answer"])
print("Kaynaklar:", response["sources"])