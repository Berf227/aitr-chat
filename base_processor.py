import os
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class BaseProcessor:
    def __init__(self, index_root: str, model_name: str, qa_prompt: str = "", google_config: dict = None):
        self.index_root = index_root
        self.model_name = model_name
        self.qa_prompt = qa_prompt
        self.google_config = google_config
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectordb = None
        self.current_category = None

    def load_documents(self, docs_path: str):
        print(f"[DEBUG] Belgeler şu klasörden yükleniyor: {docs_path}")
        
        file_paths = [os.path.join(docs_path, fname) 
                      for fname in os.listdir(docs_path) 
                      if fname.lower().endswith(".pdf") and not fname.startswith(".")]

        all_documents = []
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()

                if not documents:
                    print(f"[UYARI] {file_path} boş içerik döndürdü, atlanıyor.")
                    continue
                
                print(f"[OK] Yüklendi: {os.path.basename(file_path)} ({len(documents)} sayfa)")
                for doc in documents:
                    source_path = doc.metadata.get("source", "")
                    doc.metadata["source"] = os.path.basename(source_path)
                    if "page" in doc.metadata:
                        doc.metadata["page"] += 1
                all_documents.extend(documents)

            except Exception as e:
                print(f"[HATA] {os.path.basename(file_path)} yüklenemedi: {e}")
        print(f"[DEBUG] Toplam yüklenen belge sayısı: {len(all_documents)}")
        return all_documents

    def chunk_documents(self, documents, chunk_size: int = 1000, chunk_overlap: int = 200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        print(f"[DEBUG] Chunk sayısı: {len(chunks)}")
        return chunks

    def get_or_create_index(self, category: str, docs_path: str):
        index_path = os.path.join(self.index_root, category)

        if self.current_category == category and self.vectordb is not None:
            print("[DEBUG] Vektör deposu önbellekten kullanılıyor.")
            return self.vectordb

        if os.path.exists(index_path):
            print(f"[DEBUG] Mevcut Chroma indeksi yükleniyor: {index_path}")
            self.vectordb = Chroma(
                persist_directory=index_path,
                embedding_function=self.embeddings
            )
        else:
            print(f"[DEBUG] Yeni indeks oluşturuluyor: {index_path}")
            documents = self.load_documents(docs_path)
            chunks = self.chunk_documents(documents)

            if not chunks:
                raise ValueError("[HATA] Chunk işlemi sonucu boş. İndeks oluşturulmadı.")

            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=index_path
            )
            batch_size=1000
            for i in range(0, len(chunks), batch_size):
                self.vectordb.add_documents(chunks[i:i+batch_size])
            #self.vectordb.persist()
            print("[DEBUG] Yeni Chroma indeksi oluşturuldu ve kaydedildi.")

        self.current_category = category
        return self.vectordb

##### NEW METHODS #####
    def get_retriever(self, category: str, docs_path: str):
        vectordb = self.get_or_create_index(category, docs_path)
        retriever = vectordb.as_retriever(search_kwargs={"k": 8})
        return retriever

    def answer_question(self, question: str, category: str, docs_path: str):
        vectordb = self.get_or_create_index(category, docs_path)

        try:
            doc_count = vectordb._collection.count()
        except Exception:
            doc_count = "bilinmiyor"
        print(f"[DEBUG] Vektör deposunda yaklaşık {doc_count} belge var.")

        retriever = vectordb.as_retriever(search_kwargs={"k": 8})
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"[DEBUG] Alınan ilgili belge sayısı: {len(relevant_docs)}")

        llm = ChatGoogleGenerativeAI(
            model=self.google_config["model"],
            google_api_key=self.google_config["gemini_api_key"],
            temperature=0.5
        )

        template_text = self.qa_prompt.strip() or (
            "Kontekste göre soruyu cevapla:\n\n{context}\n\nSoru: {input}\nCevap:"
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template=template_text
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

        try:
            print(f"[DEBUG] QA zinciri çalıştırılıyor...")
            result = qa_chain.invoke({"input": question})
            print(f"[DEBUG] QA zinciri sonucu başarıyla alındı.")
        except Exception as e:
            print(f"[ERROR] QA zinciri çalıştırılamadı: {e.__class__.__name__} - {str(e)}")
            traceback.print_exc()
            return {"answer": "Bir hata oluştu. Sistem yöneticisine danışınız.", "sources": []}

        try:
            answer = result.get("result", "[Boş cevap]")
            sources = [
                f"{doc.metadata.get('source', 'Bilinmeyen')} (Sayfa {doc.metadata.get('page', '?')})"
                for doc in result.get("source_documents", [])
            ]
            print(f"[DEBUG] Answer tip: {type(answer)}")
            print(f"[DEBUG] Sources: {sources}")

        except Exception as e:
            print(f"[ERROR] LLM sonucu ayrıştırılamadı: {e}")
            return {"answer": "Yanıt alınamadı.", "sources": []}

        return {"answer": answer, "sources": sources}
