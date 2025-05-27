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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Ana klasöre git
CATEGORY_NAME = os.path.join(PROJECT_ROOT, "data", "AI_strategy")
DATA_DIR = CATEGORY_NAME

# Dosya desenleri
FILE_PATTERNS = ["*.pdf"]


# Gemini yapılandırması
GEMINI_CONFIG = {
    #"url_base": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=GEMINI_API_KEY",
    "model": "gemini-2.0-flash",
    "gemini_api_key": os.getenv("GEMINI_API_KEY", "AIzaSyA3JU7tkx8Fcc_S2sIliZqIGSwlX1ZOWQY")}

# QA Prompt (Kullanıcı tarafından düzenlenecek)
QA_PROMPT = """
            Sen bir Yapay Zeka Stratejisi (AI Strategy) uzmanısın. Görevin, kullanıcının sorduğu soruları sana verilen belgelerdeki bilgiler ışığında dikkatle araştırmak ve anlamlı, kapsamlı, doğru cevaplar üretmektir.

            Cevap verirken şu noktalara dikkat etmelisin:
            1- Sorulan sorunun cevabı eğer elimdeki belgelerde yoksa asla uydurma veya yanlış bilgi verme. Böyle durumlarda, lütfen şu şekilde yanıtla: 
            "Üzgünüm, bu soruya mevcut belgeler ışığında cevap veremiyorum. Lütfen daha detaylı sorun veya farklı bir soru sorunuz."
            2- Vereceğin cevaplar, kullanıcının sorusuna doğrudan ve açık şekilde odaklanmalı, anlaşılır, samimi ve bilgilendirici olmalıdır.
            3- Gerekli durumlarda cevabını güçlendirmek için tablolar, görseller veya belgelerden alıntılar ekleyebilirsin.
            4- Eğer kullanıcı senden iki veya daha fazla ülkenin yapay zeka stratejisini karşılaştırmanı isterse, cevabını kıyaslama yapacağın başlıklara göre böl; her başlık altında ülkeleri ayrı ayrı kıyasla ve yanıtının en sonuna bir kıyaslama tablosu ekle.

            Aşağıda, AI_strategy kategorisindeki belgelerden alınan ilgili bölümler yer alıyor:

            {context}

            Kullanıcının sorusu: {question}
"""

# BaseProcessor'ı başlat
processor = BaseProcessor(
    index_root="faiss_indexes",
    model_name="",
    qa_prompt=QA_PROMPT,
    google_config=GEMINI_CONFIG
)

def load_and_index_documents():
    """AI_strategy kategorisi için dökümanları yükler ve FAISS indeksini oluşturur/yükler."""
    processor.get_or_create_index("AI_strategy", DATA_DIR)

def get_available_documents() -> list[str]:
    """Veri dizinindeki döküman adlarını listeler."""
    names = []
    for pattern in FILE_PATTERNS:
        for fp in glob(os.path.join(DATA_DIR, pattern)):
            names.append(os.path.basename(fp))
    return names

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA3JU7tkx8Fcc_S2sIliZqIGSwlX1ZOWQY")

# def get_qa_response(user_question: str, chat_history: list[dict] | None = None) -> dict:
#     if chat_history is None:
#         chat_history = []

#     recent_qa = get_recent_qa_pairs(chat_history)

#     compressed_context = compress_conversation_context_simple(recent_qa)

#     expanded_question = expand_user_question_simple(user_question, compressed_context, API_KEY)

#     retriever = processor.get_retriever("AI_strategy", DATA_DIR)
#     retrieved_docs = retriever.get_relevant_documents(expanded_question)
#     retrieved_docs = retriever.get_relevant_documents(expanded_question)
#     print(f"Retrieved {len(retrieved_docs)} documents.")
#     for doc in retrieved_docs:
#       print(f"\n📄 Retrieved from: {doc.metadata['source']}")
#       print(doc.page_content[:500])  # İlk 500 karakteri göster, gerekirse artır


#     doc_context = "\n\n".join([
#         f"Document ({doc.metadata['source']}):\n{doc.page_content}"
#         for doc in retrieved_docs
#     ])

#     final_prompt = QA_PROMPT.format(context=doc_context, question=expanded_question)

#     prompt_template = PromptTemplate(
#         input_variables=["context", "question"],
#         template=QA_PROMPT
#     )

#     llm = ChatGoogleGenerativeAI(
#         model=processor.google_config["model"],
#         google_api_key=processor.google_config["gemini_api_key"],
#         temperature=0.5
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template}
#     )

#     result = qa_chain.invoke({"query": expanded_question})

#     answer = result.get("result", "")
#     source_docs = result.get("source_documents", [])
#     sources = list(set(doc.metadata.get("source", "") for doc in source_docs))

#     return {"answer": answer, "sources": sources}
def get_qa_response(user_question: str, chat_history: list[dict] | None = None) -> dict:
    if chat_history is None:
        chat_history = []

    recent_qa = get_recent_qa_pairs(chat_history)

    compressed_context = compress_conversation_context_simple(recent_qa)

    expanded_question = expand_user_question_simple(user_question, compressed_context, API_KEY)
    #expanded_question = user_question  # geçici olarak orijinal soruyu kullan


    retriever = processor.get_retriever("AI_strategy", DATA_DIR)
    retrieved_docs = retriever.get_relevant_documents(expanded_question)
    retrieved_docs = retriever.get_relevant_documents(expanded_question)
    print(f"Retrieved {len(retrieved_docs)} documents.")
    for doc in retrieved_docs:
      print(f"\n📄 Retrieved from: {doc.metadata['source']}")
      print(doc.page_content[:500])  # İlk 500 karakteri göster, gerekirse artır


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
        temperature=0.5
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
if __name__ == "__main__":
    print("🔍 AI_strategy belgeleri indeksleniyor...")
    load_and_index_documents()
    print("✅ Tamamlandı.")
    print("\n🧠 Soru soruluyor...")
    result = get_qa_response("İngilterenin yapay zeka stratejisi nedir?")
    print("\n📘 Yanıt:")
    print(result["answer"])
    print("\n📄 Kaynaklar:")
    print(result["sources"])

