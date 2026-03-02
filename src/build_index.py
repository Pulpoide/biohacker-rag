import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rich.console import Console

console = Console()

load_dotenv()


def load_and_chunk_document(file_path):
    """Carga el documento y lo divide en fragmentos (chunks) semánticos."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(document)


def create_vector_store(chunks, persist_db_path):
    """Genera embeddings con OpenAI y los almacena en ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_db_path
    )
    return vector_store


def main():
    console.print(
        "\n[bold cyan]🚀 Iniciando indexación del BioHacker Root...[/bold cyan]"
    )

    try:
        file_to_index = "data/faq_document.txt"

        with console.status(
            "[bold green]Procesando manual de operaciones...[/bold green]"
        ):
            chunks = load_and_chunk_document(file_to_index)

        console.print(
            f"[bold green]✅ Documento procesado:[/bold green] [yellow]{len(chunks)}[/yellow] chunks generados."
        )

        with console.status("[bold blue]Sincronizando con ChromaDB...[/bold blue]"):
            create_vector_store(chunks, "db_biohacker")

        console.print(
            "[bold magenta]💾 ChromaDB creada y persistida en 'db_biohacker/'.[/bold magenta]"
        )
        console.print(
            "[bold green]🧠 El cerebro de Root está listo para operar.[/bold green]\n"
        )

    except Exception as e:
        console.print(
            f"[bold red]❌ Error crítico durante la indexación:[/bold red] {e}"
        )


if __name__ == "__main__":
    main()
