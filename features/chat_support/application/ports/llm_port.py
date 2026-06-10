from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
from ...domain.models import ChatMessage

class LLMPort(ABC):
    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        conversation_history: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        pass

    @abstractmethod
    def generate_structured_response(
        self,
        system_prompt: str,
        conversation_history: List[ChatMessage],
        response_format: Any,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        pass