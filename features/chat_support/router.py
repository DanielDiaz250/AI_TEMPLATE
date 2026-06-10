import os
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from core.infrastructure.openai_client import CoreOpenAIClient
from core.infrastructure.chroma_client import CoreChromaClient

from .domain.models import ChatSession
from .application.use_cases import (
    ExecuteChatRAGUseCase,
    ExecuteChatRAGInput,
    ExecuteChatRAGOutput,
    EvaluateRAGResponseUseCase,
    EvaluateRAGResponseInput,
    RecordUserFeedbackUseCase,
    RecordUserFeedbackInput
)
from .adapters.openai_chat_adapter import OpenAIChatAdapter
from .adapters.chroma_chat_adapter import ChromaChatAdapter
from .adapters.langfuse_adapter import LangfuseTelemetryAdapter
from .adapters.mock_ticket_adapter import MockTicketAdapter
from langfuse import propagate_attributes

chat_router = APIRouter(prefix="/chat", tags=["Chat Support AI"])

shared_openai = CoreOpenAIClient()
shared_chroma = CoreChromaClient()

llm_adapter = OpenAIChatAdapter(core_openai=shared_openai)
vector_db_adapter = ChromaChatAdapter(core_chroma_client=shared_chroma, core_openai_client=shared_openai)
telemetry_adapter = LangfuseTelemetryAdapter()
ticket_adapter = MockTicketAdapter()

rag_use_case = ExecuteChatRAGUseCase(
    llm=llm_adapter, vector_db=vector_db_adapter, telemetry=telemetry_adapter, ticket_system=ticket_adapter
)

evaluate_use_case = EvaluateRAGResponseUseCase(
    llm=llm_adapter, telemetry=telemetry_adapter
)

feedback_use_case = RecordUserFeedbackUseCase(
    telemetry=telemetry_adapter
)

sessions_db: dict[str, ChatSession] = {}

def get_or_create_session(session_id: str, user_id: str) -> ChatSession:
    if session_id not in sessions_db:
        sessions_db[session_id] = ChatSession(session_id=session_id, user_id=user_id)
    return sessions_db[session_id]

@chat_router.post(
    "/v1/predict", 
    response_model=ExecuteChatRAGOutput, 
    status_code=status.HTTP_200_OK,
    summary="Consultar al Agente Conversacional RAG",
    description=(
        "Recibe el mensaje del usuario, recupera fragmentos pertinentes de la base de datos "
        "vectorial ChromaDB de acuerdo a la estrategia seleccionada (vector, hybrid, query_expansion, "
        "parent_child) con opción de re-ordenamiento (Reranking) de LLM, y genera una respuesta "
        "estructurada en formato Markdown.\n\n"
        "**Evaluación Asíncrona (LLM-as-a-Judge)**: Al retornar la respuesta, dispara de forma asíncrona "
        "tres evaluaciones automatizadas (Fidelidad, Relevancia de Respuesta y Relevancia del Contexto) "
        "registrando los scores en Langfuse Cloud de manera no bloqueante."
    ),
    response_description="Respuesta estructurada del agente conversacional que contiene trace_id y contexto recuperado."
)
async def chat_endpoint(payload: ExecuteChatRAGInput, background_tasks: BackgroundTasks):
    """
    Controlador HTTP de FastAPI para procesar consultas de chat técnico.
    Inicializa o recupera la sesión del usuario, propaga atributos globales a Langfuse y
    encola la evaluación de la tríada de RAG de forma asíncrona en segundo plano.
    """
    try:
        session = get_or_create_session(payload.session_id, payload.user_id)
        with propagate_attributes(
            user_id=payload.user_id,
            session_id=payload.session_id,
            metadata={
                "strategy": str(payload.strategy),
                "rerank": str(payload.rerank),
                "collection_name": str(payload.collection_name or "")
            }
        ):
            output = rag_use_case.execute(payload, session)
            
            # Lanzar la evaluación en segundo plano si hay traza activa
            if output.trace_id:
                eval_input = EvaluateRAGResponseInput(
                    trace_id=output.trace_id,
                    user_message=payload.user_message,
                    retrieved_context=output.retrieved_context or "",
                    markdown_answer=output.markdown_answer
                )
                background_tasks.add_task(evaluate_use_case.execute, eval_input)
                
            return output
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Fallo en IA.")


@chat_router.post(
    "/v1/feedback", 
    status_code=status.HTTP_200_OK,
    summary="Registrar Feedback Manual del Usuario",
    description=(
        "Permite que la interfaz de usuario (frontend) registre una calificación positiva (1 para 👍) "
        "o negativa (0 o -1 para 👎) asociada de forma inmediata a la traza identificada por trace_id "
        "en Langfuse Cloud, conviviendo directamente con las evaluaciones automáticas."
    )
)
async def user_feedback_endpoint(payload: RecordUserFeedbackInput):
    """
    Controlador HTTP de FastAPI para recibir el feedback manual del frontend.
    Invoca el caso de uso para almacenar la puntuación bajo la métrica 'user-feedback' en Langfuse.
    """
    try:
        feedback_use_case.execute(payload)
        return {"status": "success", "message": "Feedback registrado exitosamente."}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fallo al registrar feedback: {str(e)}")

