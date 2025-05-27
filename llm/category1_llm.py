import os
from glob import glob
from base_processor import BaseProcessor
from llm.chat_context_manager import (
    get_recent_qa_pairs,
    compress_conversation_context_simple,
    expand_user_question_simple,
    query_gemini_simple
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()  

# Kategori bilgileri
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATEGORY_NAME = os.path.join(PROJECT_ROOT, "data", "AI_Acts")
DATA_DIR = CATEGORY_NAME

FILE_PATTERNS = ["*.pdf"]

GEMINI_CONFIG = {
    "url_base": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=GEMINI_API_KEY",
    "model": "gemini-2.0-flash",
    "gemini_api_key": os.getenv("GEMINI_API_KEY", "AIzaSyBVRqRGWL52ArE5ERlzssE9GlrQ7xyEYrQ")
}

QA_PROMPT = """
            Sen bir Yapay Zeka Mevzuatı (AI Acts) uzmanısın. Görevin, kullanıcının sorduğu soruları sana verilen belgelerdeki bilgiler ışığında dikkatle araştırmak ve anlamlı, kapsamlı, doğru cevaplar üretmektir.

            Cevap verirken şu noktalara dikkat etmelisin:
            1- Sorulan sorunun cevabı eğer elimdeki belgelerde yoksa asla uydurma veya yanlış bilgi verme. Böyle durumlarda, lütfen şu şekilde yanıtla:
            "Üzgünüm, bu soruya mevcut belgeler ışığında cevap veremiyorum. Lütfen daha detaylı sorun veya farklı bir soru sorunuz."
            2- Vereceğin cevaplar, kullanıcının sorusuna doğrudan ve açık şekilde odaklanmalı, anlaşılır, samimi ve bilgilendirici olmalıdır.
            3- Gerekli durumlarda cevabını güçlendirmek için tablolar, görseller veya belgelerden alıntılar ekleyebilirsin.
            4- European Union (AB) belgeleri çok önemlidir. Eğer konu European Union veya REGULATION (mevzuat) ile ilgiliyse, mutlaka bu belgeler üzerinde detaylı analiz yap ve kullanıcının sorusuna en kapsamlı şekilde cevap ver.
            5- AI Acts alanında asla uydurma cevaplar verme; olabildiğince kapsamlı, net ve bilgilendirici ol.

            Aşağıda, AI_Acts kategorisindeki belgelerden alınan ilgili bölümler yer alıyor:

            {context}

            Kullanıcının sorusu: {question}
"""

processor = BaseProcessor(
    index_root="faiss_indexes",
    qa_prompt=QA_PROMPT,
    google_config=GEMINI_CONFIG
)

def load_and_index_documents():
    processor.get_or_create_index("AI_Acts", DATA_DIR)

def get_available_documents() -> list[str]:
    names = []
    for pattern in FILE_PATTERNS:
        for fp in glob(os.path.join(DATA_DIR, pattern)):
            names.append(os.path.basename(fp))
    return names


### FOLLOW UP QUESTIONS ###
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBVRqRGWL52ArE5ERlzssE9GlrQ7xyEYrQ")


def get_qa_response(user_question: str, chat_history: list[dict] | None = None) -> dict:
    if chat_history is None:
        chat_history = []

    recent_qa = get_recent_qa_pairs(chat_history)

    compressed_context = compress_conversation_context_simple(recent_qa)

    expanded_question = expand_user_question_simple(user_question, compressed_context, API_KEY)

    retriever = processor.get_retriever("AI_Acts", DATA_DIR)
    retrieved_docs = retriever.get_relevant_documents(expanded_question)

    doc_context = "\n\n".join([
        f"Document ({doc.metadata['source']}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    final_prompt = QA_PROMPT.format(context=doc_context, question=expanded_question)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT
    )

    llm = ChatGoogleGenerativeAI(
        model=processor.google_config["model"],
        google_api_key=processor.google_config["gemini_api_key"],
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa_chain.invoke({"query": expanded_question})

    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])
    sources = list(set(doc.metadata.get("source", "") for doc in source_docs))

    return {"answer": answer, "sources": sources}
