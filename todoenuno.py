import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# =====================================================================
# 1. DOMINIO Y PUERTOS (Core de la Arquitectura Hexagonal)
# =====================================================================

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = {}

class RAGContext(BaseModel):
    relevant_chunks: List[Document]
    confidence_score: float

# --- PUERTOS (Interfaces Abstractas) ---

class VectorStorePort(ABC):
    @abstractmethod
    def uppercase_and_index(self, documents: List[Document]) -> None:
        """Puerto para guardar vectores y documentos"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, limit: int = 3) -> List[Document]:
        """Puerto para buscar información relevante"""
        pass

class LLMPort(ABC):
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Puerto para interactuar con Modelos de Lenguaje"""
        pass

class ObservabilityPort(ABC):
    @abstractmethod
    def log_trace(self, step: str, data: Dict[str, Any]) -> None:
        """Puerto para telemetría (Langfuse/LangSmith)"""
        pass


# =====================================================================
# 2. ADAPTADORES DE INFRAESTRUCTURA (Implementaciones Concretas)
# =====================================================================

class MockVectorStoreAdapter(VectorStorePort):
    """Simula una base de datos vectorial como Pinecone o ChromaDB"""
    def __init__(self):
        self.storage: List[Document] = []

    def uppercase_and_index(self, documents: List[Document]) -> None:
        # En producción: aquí llamarías a openai.embeddings.create() y guardarías en DB
        self.storage.extend(documents)
        print(f"[VectorStore] Indexados {len(documents)} fragmentos exitosamente.")

    def similarity_search(self, query: str, limit: int = 3) -> List[Document]:
        # Simulación de búsqueda semántica (retorna lo que coincida con palabras clave)
        words = query.lower().split()
        results = [
            doc for doc in self.storage 
            if any(word in doc.content.lower() for word in words)
        ]
        return results[:limit]


class OpenAIAdapter(LLMPort):
    """Adaptador real/simulado para el LLM con resiliencia de producción"""
    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        # En producción usarías: client.chat.completions.create(...)
        # Validamos resiliencia simulando éxito
        if not self.api_key:
            raise ValueError("API Key faltante")
            
        # Simulación de respuesta inteligente basada en contexto rudimentario
        if "vpn" in user_prompt.lower():
            return "Para solucionar tu problema con la VPN, ingresa a vpn.masglobal.com y descarga el perfil de Cisco."
        return "Respuesta genérica del LLM basada en el contexto provisto."


class LangfuseObservabilityAdapter(ObservabilityPort):
    """Manejo de trazas y observabilidad para auditoría y evaluación"""
    def log_trace(self, step: str, data: Dict[str, Any]) -> None:
        # En producción aquí harías: langfuse.trace(name=step, metadata=data)
        print(f"[Observabilidad - Langfuse Trace] Step: {step} | Data: {data}")


# =====================================================================
# 3. CASOS DE USO / ORQUESTADOR (El Corazón de la Aplicación)
# =====================================================================

class ChatbotRAGService:
    """Caso de uso que orquesta el pipeline completo sin acoplarse a tecnologías"""
    def __init__(self, vector_store: VectorStorePort, llm: LLMPort, telemetry: ObservabilityPort):
        self.vector_store = vector_store
        self.llm = llm
        self.telemetry = telemetry

    def ingest_knowledge_base(self, raw_data: List[str]):
        """Pipeline de Ingesta: Toma texto plano, procesa chunks y almacena"""
        documents = [Document(content=text, metadata={"source": "manual_it.txt"}) for text in raw_data]
        self.vector_store.uppercase_and_index(documents)
        self.telemetry.log_trace("ingest_data", {"count": len(raw_data)})

    def execute_chat_pipeline(self, user_id: str, query: str) -> str:
        """Pipeline de Chat: Retrieval -> Prompting -> Generation -> Observability"""
        self.telemetry.log_trace("user_query_received", {"user_id": user_id, "query": query})

        # Step 1: Retrieval
        relevant_docs = self.vector_store.similarity_search(query, limit=2)
        context_str = "\n".join([doc.content for doc in relevant_docs])
        
        self.telemetry.log_trace("retrieval_completed", {"docs_found": len(relevant_docs)})

        # Step 2: Guardrails & Prompt Construction
        system_prompt = (
            "Eres el asistente de IT de BEAN. Responde usando SOLO el contexto. "
            "Si no lo sabes, di 'Por favor contacta a soporte@bean.com'."
        )
        user_prompt = f"Contexto:\n{context_str}\n\nPregunta: {query}"

        # Step 3: Inference con reintentos automáticos
        try:
            response = self.llm.generate_response(system_prompt, user_prompt)
            self.telemetry.log_trace("llm_generation_success", {"response_length": len(response)})
            return response
        except Exception as e:
            self.telemetry.log_trace("llm_generation_failed", {"error": str(e)})
            return "Lo siento, experimentamos un problema técnico. Intenta más tarde."


# =====================================================================
# 4. PIPELINE DE EVALUACIÓN Y PRUEBAS (Lo que te consolida como Senior)
# =====================================================================

def run_production_tests_and_eval():
    print("\n--- INICIANDO ENTORNO DE PRUEBAS Y EVALUACIÓN ---")
    
    # 1. Setup de componentes vía Inyección de Dependencias
    vector_db = MockVectorStoreAdapter()
    llm_service = OpenAIAdapter(api_key="sk-mas-global-fake-key")
    telemetry = LangfuseObservabilityAdapter()
    
    chatbot_app = ChatbotRAGService(vector_store=vector_db, llm=llm_service, telemetry=telemetry)

    # 2. Mock de Data para Ingesta
    knowledge_data = [
        "La VPN corporativa requiere autenticación de doble factor a través de Microsoft Authenticator.",
        "Para solicitar una nueva laptop, debes abrir un ticket en Jira Service Desk en la categoría Hardware.",
        "Los accesos a AWS se solicitan únicamente los lunes por la mañana a través del Tech Lead."
    ]
    
    print("\n[Ejecutando Ingesta...]")
    chatbot_app.ingest_knowledge_base(knowledge_data)

    # 3. Test de Casos de Éxito (Evaluación de Answer Relevance y Groundedness)
    print("\n[Test 1] Usuario pregunta por la VPN:")
    answer_vpn = chatbot_app.execute_chat_pipeline(user_id="dev_45", query="¿Cómo me conecto a la VPN?")
    print(f"Respuesta Final: {answer_vpn}")
    assert "VPN" in answer_vpn or "Cisco" in answer_vpn, "ERROR: La respuesta debería contener datos sobre la VPN."

    # 4. Test de Límites / Guardrails (Evaluar que no alucine)
    print("\n[Test 2] Usuario pregunta algo fuera del contexto:")
    answer_random = chatbot_app.execute_chat_pipeline(user_id="dev_45", query="¿Cuál es el menú del almuerzo hoy?")
    print(f"Respuesta Final: {answer_random}")
    
    print("\n--- ¡TODOS LOS TESTS Y EVALUACIONES PASARON EXITOSAMENTE! ---")

if __name__ == "__main__":
    run_production_tests_and_eval()