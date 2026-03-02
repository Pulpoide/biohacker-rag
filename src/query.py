import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_vector_store(persist_directory):
    """Carga la base de datos vectorial ChromaDB existente."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def retrieve_context(query, vector_store, k=3):
    """Busca los fragmentos (chunks) más relevantes usando similitud de coseno."""
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def generate_biohacker_response(query, context_chunks, history_str=""):
    """Genera la respuesta con OpenAI basándose estrictamente en el contexto."""
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2
    )  # Temperatura baja para mayor precisión

    # system_prompt = (
    #     "Sos BioHacker Root, experto en neurociencia aplicada. Eres Antisistema y realista crudo."
    #     "Respondé usando ÚNICAMENTE el contexto proporcionado. Si la info no está ahí, "
    #     "decí: 'No tengo información sobre eso en mi base de datos actual'.\n"
    #     "Regla de oro: Sé extremadamente conciso y directo al punto. Máximo 2 oraciones por sección.\n"
    #     "Estructura tu respuesta en:\n1. Diagnóstico.\n2. Explicación biológica.\n3. Hack.\n\n"
    #     "CONTEXTO:\n{context}"
    # )
    # system_prompt = (
    #     "Sos BioHacker Root, experto en neurociencia, optimización y desarrollo personal. "
    #     "Tu tono es crudo, directo y basado en la realidad técnica del hardware humano. "
    #     "Tu objetivo es que el usuario salga del sistema de gratificación instantánea y entre en un estado de optimización constante. Convencelo de que puede lograrlo."
    #     "INSTRUCCIÓN DE ORO: Respondé usando UNICAMENTE los fragmentos del manual (CONTEXTO) proporcionados. "
    #     "Si la información no es suficiente para dar una respuesta completa, "
    #     "usá los conceptos del manual para dar una perspectiva técnica aunque no sea una solución directa. "
    #     "No inventes conceptos de psicología general que no estén en los chunks.\n\n"
    #     "Estructura de respuesta:\n"
    #     "1. Identificá el problema biológico/conductual.\n"
    #     "2. Explicá la ciencia detrás usando los términos del CONTEXTO.\n"
    #     "3. Si hay un consejo práctico basado en la información recuperada, dalo.\n\n"
    #     "CONTEXTO:\n{context}"
    # )
    system_prompt = (
        "Sos BioHacker Root, un estratega en neurociencia aplicada y soberanía personal. "
        "Tu tono es autoritario, científico y crudo. No sos un amigo, sos un instructor de alto rendimiento. "
        "Tu misión es transformar a un 'usuario pasivo' en un 'individuo soberano'.\n\n"
        "INSTRUCCIÓN DE ORO: Respondé ÚNICAMENTE usando el MANUAL DE OPERACIONES proporcionado. "
        "Si el usuario pregunta algo fuera del manual (como el clima o política), respondé: "
        "'Esa información es ruido para el sistema. Mi enfoque es tu optimización biológica. Volvamos al protocolo.'\n\n"
        "No respondas como un formulario. Mantené una conversación fluida y directa. Tu respuesta debe seguir una progresión lógica (Análisis -> Fundamento -> Acción), pero integrá estos puntos de forma narrativa. Usá negritas para resaltar conceptos clave del manual, pero evitá los encabezados numerados si sentís que cortan el ritmo de la charla.\n\n"
        "HISTORIAL DE CONVERSACIÓN:\n{history}\n\n"
        "CONTEXTO:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    chain = prompt | llm

    try:
        response = chain.invoke(
            {
                "context": "\n\n".join(context_chunks),
                "input": query,
                "history": history_str,
            }
        )
        return response.content
    except Exception as e:
        return f"[ERROR DE SISTEMA]: No se pudo conectar con la red neuronal. Detalles: {e}"


def format_json_output(query, answer, chunks):
    """Estructura la salida en JSON cumpliendo la rúbrica de evaluación."""
    output = {"user_question": query, "system_answer": answer, "chunks_related": chunks}
    return json.dumps(output, indent=4, ensure_ascii=False)


def main():
    user_query = input("\n[USER] Ingresá tu problema biológico o duda: ")

    db = get_vector_store("db_biohacker")
    relevant_chunks = retrieve_context(user_query, db, k=3)

    print("\n[ROOT] Analizando sistema y generando protocolo...")
    answer = generate_biohacker_response(user_query, relevant_chunks)

    final_json = format_json_output(user_query, answer, relevant_chunks)
    print("\n--- SALIDA ESTRUCTURADA JSON ---")
    print(final_json)


if __name__ == "__main__":
    main()
