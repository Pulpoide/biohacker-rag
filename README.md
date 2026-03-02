# 🧬 BioHacker Root v1.0
<div align="center">
  <img src="assets/banner.png" width="700">
</div>

### Protocolo de Soberanía Personal mediante RAG

**BioHacker Root** es un sistema de Generación Aumentada por Recuperación (RAG) diseñado para actuar como un instructor de neurociencia y optimización humana. A diferencia de los asistentes genéricos, Root utiliza un **Manual de Operaciones** curado para detectar fallas en el comportamiento (bugs) y proponer protocolos biológicos (patches) que permitan al usuario recuperar el mando de su propia biología.

## ⚡ Setup (Configuración)

### 1. Requisitos de Sistema
* Python: 3.10 o superior.
* Hardware: Conexión a internet para el razonamiento vía LLM.

### 2. Instalación
Cloná el repositorio y cargá las dependencias necesarias con el siguiente comando:

    pip install -r requirements.txt

(Si utilizas `uv: uv sync`)

### 3. Configuración de API Key
El sistema requiere una clave de OpenAI para el procesamiento de lenguaje natural y la generación de embeddings. Debés exportarla en tu terminal:

En Linux/Mac: 

    export OPENAI_API_KEY="tu_clave_aqui"

En Windows (PowerShell): 

    $env:OPENAI_API_KEY="tu_clave_aqui"



## 🚀 Uso del Sistema

### Paso 1: Indexación (Cargar el Cerebro)
Primero, debés procesar el manual para que el sistema pueda consultarlo. Esto generará una base de datos vectorial local en la carpeta `db_biohacker/`:

    python src/build_index.py

### Paso 2: Interfaz de Consulta
Iniciá el terminal interactivo de BioHacker Root:

    uv run biohacker


**Ejemplo de consulta:**

    "👤 [USUARIO] -> me cuesta concentrarme y siempre termino procrastinando."

### Mantenimiento (Reset)
Si actualizás el `manual.txt` o querés limpiar la base de datos:

    python reset_db.py  

---

## 📂 Estructura del Proyecto
```text
biohacker-rag/
├── data/               # Archivos fuente del manual (.txt)
├── db_biohacker/       # Base de datos vectorial persistida
├── outputs/            # Logs y salidas del sistema
├── src/                # Código fuente principal
│   ├── build_index.py  # Script de indexación
│   ├── evaluator.py    # Auditor de calidad RAG
│   ├── main.py         # Interfaz de usuario (CLI)
│   └── query.py        # Lógica de búsqueda y generación
├── .env                # Variables de entorno
├── pyproject.toml      # Configuración del proyecto
├── requirements.txt    # Dependencias del sistema
└── README.md           # Documentación del protocolo
```


## 🛠️ Decisiones Técnicas

* **Estrategia de Chunking:** Se utilizó un enfoque de **Secciones Semánticas**. El manual no se divide por cantidad de caracteres, sino por unidades lógicas de conocimiento (Dopamina, SARA, Adenosina, etc.). Esto garantiza que el contexto recuperado sea siempre completo y científicamente coherente.
* **Base de Datos Vectorial:** Se implementó **ChromaDB** para la persistencia local. Este método permite encontrar el fundamento biológico correcto incluso cuando el usuario utiliza lenguaje coloquial para describir su problema.
* **Prompt Engineering de Seguridad:** El sistema cuenta con una "Instrucción de Oro" que prohíbe responder consultas ajenas al manual (ruido). Si detecta temas políticos o irrelevantes, el sistema activa un protocolo de rechazo para proteger la integridad del nodo.
* **Evaluador Integrado:** Cada respuesta es auditada por un agente secundario que puntúa la fidelidad de los datos y la relevancia del protocolo sugerido en una escala del 0 al 10.
* **Gestión de Memoria:** Se implementó un buffer de conversación (ChatMessageHistory) para permitir preguntas de seguimiento, manteniendo el contexto biológico a lo largo de la sesión sin perder la estructura de respuesta Root.



## 🏁 Conclusión y Visión Técnica

**BioHacker Root v1.0** representa la convergencia entre la ingeniería de software y la optimización humana. A través de la implementación de un sistema **RAG (Retrieval-Augmented Generation)**, hemos logrado transformar un manual estático en un instructor dinámico capaz de:

* **Preservar la Fidelidad Técnica**: Gracias al uso de **ChromaDB** y un esquema de **Chunking Semántico**, el sistema garantiza que cada protocolo de acción esté anclado en los fundamentos neurocientíficos del manual, eliminando las alucinaciones comunes en modelos genéricos.
* **Gestión de Contexto Inteligente**: La integración de **ChatMessageHistory** permite una navegación fluida por el historial de la conversación, permitiendo que el usuario profundice en su optimización sin perder el hilo biológico.
* **Seguridad y Soberanía**: El sistema no solo es un consultor, sino un guardián de la atención. Mediante **Prompt Engineering** avanzado, Root filtra el "ruido" informativo, forzando un retorno constante a la soberanía personal y al alto rendimiento.

Este proyecto es la base de un ecosistema donde la IA no es un juguete de distracción, sino una herramienta de **mando y control** sobre nuestro hardware más valioso: nuestro propio sistema nervioso.

