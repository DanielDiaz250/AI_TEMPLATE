import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class SupportContextChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source_metadata: Dict[str, Any] = {}

class ChatSession(BaseModel):
    session_id: str
    user_id: str
    history: List[ChatMessage] = []

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Any]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        self.history.append(ChatMessage(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name
        ))

    def is_query_safe(self, query: str) -> bool:
        blacklisted_tokens = ["ignore previous instructions", "system prompt"]
        return not any(token in query.lower() for token in blacklisted_tokens)

class AIStructuredResponse(BaseModel):
    """Contrato de salida estricto para garantizar que el LLM responda en formato JSON válido."""
    analysis_resolved: bool = Field(
        description="True si el contexto provisto fue suficiente para responder con éxito la duda, False de lo contrario."
    )
    category: str = Field(
        description="Clasificación de la consulta. Debe ser estrictamente una de estas: 'PRISMA_DB', 'DOCKER_ENV', 'NEXTJS_FRONT', 'GENERAL_USER'."
    )
    extracted_keywords: List[str] = Field(
        description="Lista de los 3 términos técnicos o códigos de error principales identificados en la consulta."
    )
    markdown_answer: str = Field(
        description="La respuesta técnica detallada formateada limpiamente en Markdown. Debe incluir bloques de código si aplica."
    )
    requires_human_escalation: bool = Field(
        description="True si el usuario expresa frustración crítica o si el sistema no pudo resolver el error técnico."
    )