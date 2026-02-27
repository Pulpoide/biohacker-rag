import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def load_and_chunk_document(file_path):
    """Carga el documento y lo divide en fragmentos (chunks)."""
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    # Estrategia: Tamaño fijo con solapamiento (Overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )
    return text_splitter.split_documents(document)


def create_vector_store(chunks, persist_db_path):
    """Genera embeddings y los almacena en ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Creamos la base de datos y la persistimos en el disco
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_db_path
    )
    return vector_store


def main():
    print("🚀 Iniciando indexación del BioHacker Root...")

    # 1. Cargar y procesar
    chunks = load_and_chunk_document("data/faq_document.txt")
    print(f"✅ Documento procesado: {len(chunks)} chunks generados.")

    # 2. Indexar en ChromaDB
    create_vector_store(chunks, "db_biohacker")
    print("💾 ChromaDB creada y persistida en 'db_biohacker/'.")


if __name__ == "__main__":
    main()
