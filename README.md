# BEAN - Life Intelligence & IT Support Platform 🚀

Bienvenido a la **Plataforma de IA Corporativa de BEAN**. Esta aplicación está diseñada bajo principios de **Clean Architecture (Arquitectura Hexagonal)** en Python, exponiendo endpoints robustos con FastAPI y brindando soporte técnico RAG interactivo con un flujo completo de **Observabilidad y Evaluaciones (LLM-as-a-Judge)** integrado con **Langfuse Cloud**.

---

## 🏗️ Estructura del Proyecto (Arquitectura Hexagonal)

El código sigue estrictamente la separación de responsabilidades para aislar las reglas de negocio de la infraestructura:

```text
features/chat_support/
├── domain/                      <-- CAPA DE DOMINIO: Modelos de negocio puros (ej. ChatSession, ChatMessage)
│   └── models.py
├── application/                 <-- CAPA DE APLICACIÓN: Reglas de negocio de la aplicación
│   ├── ports/                   <-- PUERTOS: Interfaces abstractas (contratos)
│   │   ├── llm_port.py
│   │   ├── telemetry_port.py
│   │   └── vector_store_port.py
│   ├── use_cases/               <-- CASOS DE USO: Orquestación del flujo de la app
│   │   ├── chat_rag.py          <-- Agente conversacional RAG
│   │   ├── evaluate_rag.py      <-- Juez de evaluación (Fidelidad, Relevancia)
│   │   └── record_feedback.py   <-- Registro de feedback manual del usuario
│   └── tools/                   <-- HERRAMIENTAS: Búsqueda y creación de tickets (Tool Calling)
└── adapters/                    <-- CAPA DE INFRAESTRUCTURA: Implementaciones de adaptadores concretos
    ├── openai_chat_adapter.py   <-- Adaptador concreto para OpenAI GPT-4o
    ├── chroma_chat_adapter.py   <-- Adaptador para búsquedas en ChromaDB
    └── langfuse_adapter.py      <-- Adaptador para observabilidad y scores en Langfuse
```

---

## 🛠️ Guía de Configuración e Inicio Rápido (Para Juniors)

Sigue estos sencillos pasos para levantar la plataforma en tu entorno local:

### 1. Requisitos Previos
*   Python 3.10 o superior instalado.
*   NVM para Windows (si necesitas manejar versiones de Node.js en módulos frontend).

### 2. Creación del Entorno Virtual e Instalación
Abre la consola en el directorio raíz del proyecto y ejecuta:

```powershell
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno virtual
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configuración de Variables de Entorno (`.env`)
Crea un archivo llamado `.env` en la raíz del proyecto con la siguiente estructura:

```env
# Clave API de OpenAI
OPENAI_API_KEY=tu-clave-api-aqui
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Credenciales de Langfuse Cloud (https://cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 4. Ejecutar el Servidor de Desarrollo
Levanta la API de FastAPI localmente con recarga automática:

```powershell
uvicorn main:app --reload
```

El servidor estará disponible en [http://127.0.0.1:8000](http://127.0.0.1:8000). Puedes interactuar con la documentación auto-generada entrando a:
*   **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   **Redoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 Endpoints Clave de la API

### 1. Chat Inteligente RAG (`POST /api/chat/v1/predict`)
Envía una consulta al agente técnico de IT. La respuesta final se genera mediante búsqueda híbrida, reordenamiento (rerank) y llamadas autónomas a herramientas.

*   **Payload del Request:**
    ```json
    {
      "session_id": "session-user-101",
      "user_id": "user-corp-999",
      "user_message": "¿Cómo creo una rama en el árbol?",
      "collection_name": "user_bean_recursive",
      "strategy": "hybrid",
      "rerank": true
    }
    ```
*   **Ejemplo de Petición cURL:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/api/chat/v1/predict" \
         -H "Content-Type: application/json" \
         -d '{"session_id":"s1","user_id":"u1","user_message":"¿Qué es BEAN?","strategy":"hybrid","rerank":true}'
    ```
*   **Response del API (200 OK):**
    ```json
    {
      "analysis_resolved": true,
      "category": "GENERAL_USER",
      "extracted_keywords": ["plataforma BEAN", "trayectoria de vida"],
      "markdown_answer": "La plataforma **BEAN** es una herramienta inteligente...",
      "requires_human_escalation": false,
      "sources_used": ["user_manual"],
      "retrieved_context": "### Paso 1: Onboarding...",
      "trace_id": "a0f94a8416132bf190ef19d9d7bd7e14"
    }
    ```

### 2. Feedback de Usuario (`POST /api/chat/v1/feedback`)
Registra la calificación manual 👍/👎 del usuario sobre una respuesta específica.

*   **Payload del Request:**
    ```json
    {
      "trace_id": "a0f94a8416132bf190ef19d9d7bd7e14",
      "value": 1,
      "comment": "Respuesta muy precisa, ¡gracias!"
    }
    ```
*   **Ejemplo de Petición cURL:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/api/chat/v1/feedback" \
         -H "Content-Type: application/json" \
         -d '{"trace_id":"a0f94a8416132bf190ef19d9d7bd7e14","value":1,"comment":"Excelente"}'
    ```

---

## 📈 Observabilidad y La Tríada de RAG

El sistema de observabilidad está diseñado para medir de forma autónoma e invisible para el cliente la calidad de nuestro pipeline RAG.

Cada vez que el endpoint `/v1/predict` devuelve una respuesta, FastAPI dispara una **tarea en segundo plano (BackgroundTask)** que ejecuta de forma no bloqueante el evaluador **LLM-as-a-Judge**:

1.  **Fidelidad (Faithfulness)**: Compara la respuesta contra el contexto recuperado de ChromaDB. Asigna una calificación de 1 a 5 detectando alucinaciones.
2.  **Relevancia de Respuesta (Answer Relevance)**: Compara la respuesta contra la pregunta original. Asigna una calificación de 1 a 5 midiendo la utilidad directa.
3.  **Relevancia del Contexto (Context Relevance)**: Compara la pregunta contra el contexto recuperado. Califica si el buscador trajo información relevante o ruido.

Tanto los **3 scores automáticos** como el **feedback manual del usuario** se agrupan bajo la misma traza (`trace_id`) en tu panel de Langfuse Cloud para auditoría en vivo.
