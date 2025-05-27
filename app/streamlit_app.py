import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import time
from chat_history_manager import add_to_history, get_history
from llm.category1_llm import load_and_index_documents as load_idx_cat1, get_qa_response as qa_cat1
from llm.category2_llm import load_and_index_documents as load_idx_cat2, get_qa_response as qa_cat2
from llm.category3_llm import load_and_index_documents as load_idx_cat3, get_qa_response as qa_cat3

st.title("📚 Doküman Tabanlı Soru-Cevap Sistemi")

CATEGORY_MODULES = {
    "AI_ACT": {"load_index": load_idx_cat1, "get_qa_response": qa_cat1},
    "AI_ETHICS": {"load_index": load_idx_cat2, "get_qa_response": qa_cat2},
    "AI_STRATEGY": {"load_index": load_idx_cat3, "get_qa_response": qa_cat3},
}

if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "prev_category" not in st.session_state:
    st.session_state.prev_category = None

# 1. Kategori Seçimi
selected_category_name = st.selectbox("Kategori seçin:", list(CATEGORY_MODULES.keys()))
module = CATEGORY_MODULES[selected_category_name]

# Kategori değiştiyse indekslenme durumunu sıfırla
if st.session_state.prev_category != selected_category_name:
    st.session_state.loaded = False
    st.session_state.prev_category = selected_category_name
    get_history().clear()

# 2. Otomatik İndeksleme (tüm dökümanlar)
if not st.session_state.loaded:
    with st.spinner("Belgeler yükleniyor ve indeksleniyor…"):
        module["load_index"]()
    st.success("✅ Belgeler başarıyla indekslendi.")
    st.session_state.loaded = True

# 3. Chat Arayüzü
if st.session_state.loaded:
    for msg in get_history():
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Sorunuz:")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        add_to_history("user", user_input)

        history = get_history()
        qa_pairs = []
        question = None
        for msg in history:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant" and question is not None:
                qa_pairs.append({"question": question, "answer": msg["content"]})
                question = None

        with st.spinner("Yanıt hazırlanıyor…"):
            response = module["get_qa_response"](
                user_question=user_input,
                chat_history=qa_pairs
            )
            time.sleep(0.3)

        answer = response.get("answer", response if isinstance(response, str) else "")
        with st.chat_message("assistant"):
            st.markdown(answer)
        add_to_history("assistant", answer)
