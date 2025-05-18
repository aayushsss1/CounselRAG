import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = os.path.join(os.getcwd(), "faiss_index")

def get_text_chunks(folder_name):
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    for root, _, files in os.walk(folder_name):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            chunks = splitter.split_documents(docs)
            source_chunks.extend(chunks)
    print(f"[INFO] Total chunks created: {len(source_chunks)}")
    return source_chunks


def process_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

def persist_data(folder_name):
    chunks = get_text_chunks(folder_name)
    db = process_chunks(chunks)
    db.save_local(FAISS_INDEX_PATH)