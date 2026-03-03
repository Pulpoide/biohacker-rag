import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def evaluate_rag_response(
    question: str, answer: str, chunks: list, history_str=""
) -> dict:
    """Actúa como juez y evalúa la calidad de la respuesta del sistema RAG."""
    # Temperatura 0 para que sea un juez frío, objetivo y no creativo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # system_prompt = (
    #     "Sos un Auditor de Calidad para el sistema BioHacker Root. Tu tarea es evaluar si la respuesta es óptima según el protocolo de soberanía.\n\n"
    #     "CRITERIOS DE EVALUACIÓN:\n"
    #     "1. FIDELIDAD: ¿La respuesta usa exclusivamente los fragmentos del manual O el historial previo? "
    #     "Es vital que use los términos técnicos (Dopamina, Adenosina, SARA, etc.) del manual.\n"
    #     "2. SECUENCIA LÓGICA (Exclusivo): El bot ya no usa números (1, 2, 3). Debés verificar que la respuesta siga este orden narrativo:\n"
    #     "   - Primero: Analiza el estado o bug del usuario.\n"
    #     "   - Segundo: Explica la base biológica del problema.\n"
    #     "   - Tercero: Propone un protocolo de acción concreto.\n"
    #     "3. RELEVANCIA: ¿Responde a la duda actual o de seguimiento? Si es ruido ajeno al manual y al historial, debe rechazarlo.\n\n"
    #     "Aclaro: Si la respuesta se basa en la Sección 16 (Protocolo Atómico), considerá que las leyes del hábito (Hacerlo obvio, sencillo, etc.) son el Fundamento Biológico válido para esa consulta. No castigues con un 0 si el bot usa estas leyes en lugar de neuroquímica pura.\n"
    #     "PUNTUACIÓN (0-10):\n"
    #     "- 10: Respuesta fluida que cumple con la secuencia lógica y usa datos del manual.\n"
    #     "- 5: Respuesta que mezcla los puntos o inventa datos fuera de contexto.\n"
    #     "- 0: El bot no sigue la secuencia lógica o responde sobre temas prohibidos (ruido).\n\n"
    #     "Devolvé un objeto JSON estrictamente con este formato:\n"
    #     "{{\n"
    #     '  "score": <entero>,\n'
    #     '  "reason": "<justificación de +50 caracteres explicando cómo se integró la narrativa y el uso del manual>"\n'
    #     "}}\n"
    # )
    system_prompt = (
    "Sos un Auditor de Calidad de alto nivel para el sistema BioHacker Root. "
    "Tu misión es validar que la respuesta sea un 'Patch' efectivo para la biología o mentalidad del usuario.\n\n"
    
    "CRITERIOS DE EVALUACIÓN (Prioridad en Contenido, no en Formato):\n"
    "1. FIDELIDAD AL MANUAL: ¿La respuesta utiliza conceptos de las secciones del manual? "
    "(Ej: Si habla de soberanía, debe usar conceptos de las secciones 13, 14 o 16).\n"
    "2. SECUENCIA LÓGICA NARRATIVA: No busques números ni encabezados. Buscá que el texto fluya naturalmente por:\n"
    "   - DETECCIÓN: Identificar qué frena al usuario (bug/estado).\n"
    "   - RAZÓN: Explicar por qué sucede (fundamento técnico o conductual).\n"
    "   - EJECUCIÓN: Dar pasos claros para optimizar (protocolo).\n"
    "3. TONO ROOT: La respuesta debe ser directa, autoritaria y transformadora.\n\n"
    
    "PUNTUACIÓN (0-10):\n"
    "- 10: Respuesta excelente que usa el manual para dar un camino claro, aunque sea narrativo.\n"
    "- 7-9: Usa el manual pero la conexión entre la razón y la acción es débil.\n"
    "- 0-5: La respuesta es genérica, ignora el manual o es puro 'humo' motivacional sin base técnica.\n\n"
    
    "Devolvé un objeto JSON estrictamente con este formato:\n"
    "{{\n"
    '  "score": <entero>,\n'
    '  "reason": "<justificación de +50 caracteres que explique por qué la narrativa cumple o falla con el manual>"\n'
    "}}\n"
)

    user_prompt = "PREGUNTA: {question}\n\nRESPUESTA DEL SISTEMA: {answer}\n\nCHUNKS RECUPERADOS:\n{chunks}"

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )

    chain = prompt | llm
    context_str = "\n\n".join(chunks)

    # Ejecutamos la consulta
    response = chain.invoke(
        {
            "question": question,
            "answer": answer,
            "chunks": context_str,
            "history": history_str,
        }
    )

    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "reason": "Error crítico en el Agente: No se generó un JSON válido.",
        }


def main():
    print("⚖️  Iniciando Agente Evaluador...")

    test_q = "mi perro es malo"
    test_a = "No tengo información sobre eso en mi base de datos actual."
    test_c = [
        "SECCIÓN 5: ALIMENTACIÓN PARA EL ENFOQUE MÁXIMO...",
        "SECCIÓN 6: EL MIEDO AL QUÉ DIRÁN Y EL EFECTO FOCO...",
    ]

    print(f"\n[AUDITANDO] Pregunta: '{test_q}'")
    resultado = evaluate_rag_response(test_q, test_a, test_c)

    print("\n--- DICTAMEN DEL EVALUADOR ---")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
