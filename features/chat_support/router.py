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

@chat_router.post("/v1/predict", response_model=ExecuteChatRAGOutput, status_code=status.HTTP_200_OK)
async def chat_endpoint(payload: ExecuteChatRAGInput, background_tasks: BackgroundTasks):
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


@chat_router.post("/v1/feedback", status_code=status.HTTP_200_OK)
async def user_feedback_endpoint(payload: RecordUserFeedbackInput):
    try:
        feedback_use_case.execute(payload)
        return {"status": "success", "message": "Feedback registrado exitosamente."}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fallo al registrar feedback: {str(e)}")
