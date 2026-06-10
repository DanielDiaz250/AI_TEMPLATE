from typing import Optional
from pydantic import BaseModel, Field
from langfuse import observe
from ..ports.llm_port import LLMPort
from ..ports.telemetry_port import TelemetryPort
from ...domain.models import ChatMessage


class EvaluateRAGResponseInput(BaseModel):
    trace_id: str
    user_message: str
    retrieved_context: str
    markdown_answer: str


class JudgeEvaluation(BaseModel):
    score: int = Field(..., description="Calificación entera del 1 al 5 según la rúbrica.")
    reasoning: str = Field(..., description="Explicación detallada y paso a paso de por qué se asignó esta calificación.")


class EvaluateRAGResponseUseCase:
    """
    Caso de uso para evaluar de manera autónoma las respuestas generadas por el RAG (LLM-as-a-Judge).
    Ejecuta tres jueces independientes para calificar la Fidelidad, la Relevancia de la Respuesta
    y la Relevancia del Contexto, registrando los resultados en la nube de Langfuse.
    """
    def __init__(self, llm: LLMPort, telemetry: TelemetryPort):
        """
        Inicializa el caso de uso del evaluador inyectando sus puertos.

        Args:
            llm (LLMPort): Adaptador para realizar llamadas de evaluación al modelo juez.
            telemetry (TelemetryPort): Adaptador para guardar las puntuaciones de la evaluación.
        """
        self.llm = llm
        self.telemetry = telemetry

    @observe(name="llm-as-a-judge")
    def execute(self, input_data: EvaluateRAGResponseInput) -> None:
        """
        Orquesta y ejecuta secuencialmente las llamadas de evaluación de los tres jueces asíncronos.

        Args:
            input_data (EvaluateRAGResponseInput): Datos conteniendo la pregunta, contexto, respuesta y ID de traza.
        """
        if not input_data.trace_id:
            print("[Evaluador] Omisión: Falta trace_id.")
            return

        # Si el contexto está vacío, lo aclaramos para el evaluador de fidelidad
        contexto_para_juez = input_data.retrieved_context.strip()
        if not contexto_para_juez:
            contexto_para_juez = "(No se recuperó ningún fragmento de información de la base de conocimiento para esta consulta)"

        # 1. Evaluar fidelidad (Faithfulness)
        system_prompt_faith = (
            "Eres un evaluador experto en sistemas RAG (LLM-as-a-Judge).\n"
            "Tu tarea es calificar la 'Fidelidad' (Faithfulness) de la respuesta del asistente en base al contexto recuperado.\n"
            "Mide si la respuesta se basa ÚNICAMENTE en el contexto proporcionado y si no alucina ni inventa información técnica.\n"
            "Califica la fidelidad en una escala entera del 1 al 5:\n"
            "5: Completamente basada en el contexto, cero alucinaciones.\n"
            "4: Mayormente basada en el contexto, con alguna aclaración menor inocua.\n"
            "3: Parcialmente sustentada por el contexto, pero incluye afirmaciones que no se pueden verificar.\n"
            "2: Mayormente alucinada o inventa afirmaciones técnicas no provistas.\n"
            "1: Totalmente inventada o contradice directamente el contexto.\n\n"
            "Responde estructuradamente con la calificación y razonamiento detallado."
        )
        user_prompt_faith = [
            ChatMessage(role="user", content=f"Contexto recuperado:\n{contexto_para_juez}\n\nRespuesta generada:\n{input_data.markdown_answer}")
        ]
        
        try:
            print(f"[Evaluador] Iniciando evaluación de fidelidad para trace_id={input_data.trace_id}...")
            faith_eval = self.llm.generate_structured_response(
                system_prompt=system_prompt_faith,
                conversation_history=user_prompt_faith,
                response_format=JudgeEvaluation
            )
            self.telemetry.record_evaluation_score(
                trace_id=input_data.trace_id,
                name="fidelidad-rag",
                value=float(faith_eval.score),
                comment=f"Juez: {faith_eval.reasoning}"
            )
        except Exception as e:
            print(f"[Evaluador - Error] Fallo al evaluar fidelidad: {e}")

        # 2. Evaluar relevancia (Answer Relevance)
        system_prompt_rel = (
            "Eres un evaluador experto en sistemas RAG (LLM-as-a-Judge).\n"
            "Tu tarea es calificar la 'Relevancia de la Respuesta' (Answer Relevance) del asistente en relación con la pregunta del usuario.\n"
            "Mide si la respuesta generada responde directamente y de manera útil la pregunta original del usuario.\n"
            "Califica la relevancia en una escala entera del 1 al 5:\n"
            "5: Responde de manera exacta, clara y directa a toda la pregunta.\n"
            "4: Responde a la pregunta, pero podría ser un poco más concisa o estructurada.\n"
            "3: Responde parcialmente, dejando de lado detalles importantes de la pregunta.\n"
            "2: Mayormente irrelevante, evade responder o habla de otro tema.\n"
            "1: Totalmente irrelevante o inservible.\n\n"
            "Responde estructuradamente con la calificación y razonamiento detallado."
        )
        user_prompt_rel = [
            ChatMessage(role="user", content=f"Pregunta del usuario: {input_data.user_message}\n\nRespuesta generada:\n{input_data.markdown_answer}")
        ]
        
        try:
            print(f"[Evaluador] Iniciando evaluación de relevancia para trace_id={input_data.trace_id}...")
            rel_eval = self.llm.generate_structured_response(
                system_prompt=system_prompt_rel,
                conversation_history=user_prompt_rel,
                response_format=JudgeEvaluation
            )
            self.telemetry.record_evaluation_score(
                trace_id=input_data.trace_id,
                name="relevancia-rag",
                value=float(rel_eval.score),
                comment=f"Juez: {rel_eval.reasoning}"
            )
        except Exception as e:
            print(f"[Evaluador - Error] Fallo al evaluar relevancia: {e}")

        # 3. Evaluar relevancia del contexto (Context Relevance)
        system_prompt_context = (
            "Eres un evaluador experto en sistemas RAG (LLM-as-a-Judge).\n"
            "Tu tarea es calificar la 'Relevancia del Contexto' (Context Relevance) recuperado por el sistema RAG en relación con la pregunta del usuario.\n"
            "Mide si la información recuperada de la base de conocimiento contiene la información necesaria y pertinente para responder a la pregunta del usuario.\n"
            "Califica la relevancia del contexto en una escala entera del 1 al 5:\n"
            "5: Contiene información exacta, directa y completa que responde perfectamente la pregunta del usuario.\n"
            "4: Contiene información útil que aborda el problema del usuario, pero carece de algún detalle menor secundario.\n"
            "3: Contiene información parcialmente relevante, que ofrece contexto útil pero requiere inferencia o carece de partes importantes para responder de forma definitiva.\n"
            "2: Contiene información mayormente irrelevante que no ayuda a solucionar la consulta específica del usuario.\n"
            "1: Completamente irrelevante o inútil para la consulta del usuario.\n\n"
            "Responde estructuradamente con la calificación y razonamiento detallado."
        )
        user_prompt_context = [
            ChatMessage(role="user", content=f"Pregunta del usuario: {input_data.user_message}\n\nContexto recuperado:\n{contexto_para_juez}")
        ]
        
        try:
            print(f"[Evaluador] Iniciando evaluación de relevancia del contexto para trace_id={input_data.trace_id}...")
            context_eval = self.llm.generate_structured_response(
                system_prompt=system_prompt_context,
                conversation_history=user_prompt_context,
                response_format=JudgeEvaluation
            )
            self.telemetry.record_evaluation_score(
                trace_id=input_data.trace_id,
                name="relevancia-contexto-rag",
                value=float(context_eval.score),
                comment=f"Juez: {context_eval.reasoning}"
            )
        except Exception as e:
            print(f"[Evaluador - Error] Fallo al evaluar relevancia del contexto: {e}")

