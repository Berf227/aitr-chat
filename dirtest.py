from langchain_community.document_loaders import PyPDFLoader

file_path = "C:/Users/emree/Desktop/AI_TR/data/AI_Acts/Artificial-Intelligence-Act.pdf"
loader = PyPDFLoader(file_path)

try:
    docs = loader.load()
    print(f"{file_path} yüklendi, toplam {len(docs)} sayfa bulundu.")
except Exception as e:
    print(f"[HATA] {file_path} yüklenemedi: {e}")
