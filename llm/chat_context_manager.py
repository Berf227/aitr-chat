import requests
import os

API_KEY = os.getenv("GEMINI_API_KEY", "")

def query_gemini_simple(prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    text = ""
    if "candidates" in result:
        for candidate in result["candidates"]:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                text += part.get("text", "")
    return text

# Son 5 soru-cevap çiftini seç
def get_recent_qa_pairs(chat_history, max_tokens=1500, max_pairs=5):
    selected_pairs = []
    total_tokens = 0
    
    for pair in reversed(chat_history):
        q_tokens = len(pair.get("question", "").split())
        a_tokens = len(pair.get("answer", "").split())
        pair_tokens = q_tokens + a_tokens

        if total_tokens + pair_tokens > max_tokens:
            break

        selected_pairs.insert(0, pair)
        total_tokens += pair_tokens

        if len(selected_pairs) >= max_pairs:
            break

    return selected_pairs

def compress_conversation_context_simple(qa_pairs):
    context = ""
    for pair in qa_pairs:
        context += f"Soru: {pair['question']}\nCevap: {pair['answer']}\n"
    return context

def expand_user_question_simple(original_question, compressed_context, api_key):
    prompt = (
        f"Önceki sohbet bağlamı:\n{compressed_context}\n\n"
        f"Kullanıcının yeni sorusu:\n{original_question}\n\n"
        "Yukarıdaki bağlamı dikkate alarak, yeni soruyu daha kapsamlı, açık ve anlaşılır hale getir."
    )
    return query_gemini_simple(prompt, api_key)