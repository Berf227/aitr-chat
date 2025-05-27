from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm.category1_llm import get_qa_response as qa_cat1
from llm.category2_llm import get_qa_response as qa_cat2
from llm.category3_llm import get_qa_response as qa_cat3

app = FastAPI()

CATEGORY_MODULES = {
    "AI_ACT": qa_cat1,
    "AI_ETHICS": qa_cat2,
    "AI_STRATEGY": qa_cat3
}

class QARequest(BaseModel):
    category: str
    prompt: str
    chat_history: list[dict] = None  

from fastapi.responses import JSONResponse

@app.post("/ask")
async def ask_question(req: QARequest):
    category_key = req.category.upper()
    if category_key not in CATEGORY_MODULES:
        return JSONResponse(status_code=400, content={
            "answer": "Geçersiz kategori. 'AI_ACT', 'AI_ETHICS' veya 'AI_STRATEGY' olmalı.",
            "sources": []
        })
    
    
    if req.chat_history is None:
        req.chat_history = []

    qa_function = CATEGORY_MODULES[category_key]
    try:
        result = qa_function(user_question=req.prompt, chat_history=req.chat_history)
        return JSONResponse(content={
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        })
    except Exception as e:
        print(f"[API HATA] get_qa_response hatası: {e}")
        return JSONResponse(status_code=500, content={
            "answer": "Bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
            "sources": [],
            "error": str(e)
        })
