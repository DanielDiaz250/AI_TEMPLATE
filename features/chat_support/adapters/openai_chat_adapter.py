from typing import List, Any, Optional, Dict
from ..application.ports.llm_port import LLMPort
from ..domain.models import ChatMessage
from core.infrastructure.openai_client import CoreOpenAIClient

class OpenAIChatAdapter(LLMPort):
    def __init__(self, core_openai: CoreOpenAIClient):
        self.core_openai = core_openai

    def _convert_history(self, conversation_history: List[ChatMessage]) -> List[Dict[str, Any]]:
        raw_messages = []
        for msg in conversation_history:
            d = {"role": msg.role}
            # Evitar enviar content=None si es asistente y tiene tool_calls
            if msg.content is not None:
                d["content"] = msg.content
            elif msg.role != "assistant" or not msg.tool_calls:
                d["content"] = ""
            
            if msg.tool_calls is not None:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name is not None:
                d["name"] = msg.name
            raw_messages.append(d)
        return raw_messages

    def generate_response(
        self,
        system_prompt: str,
        conversation_history: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        raw_messages = self._convert_history(conversation_history)
        return self.core_openai.execute_chat_completion(system_prompt, raw_messages, tools=tools)

    def generate_structured_response(
        self,
        system_prompt: str,
        conversation_history: List[ChatMessage],
        response_format: Any,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        raw_messages = self._convert_history(conversation_history)
        return self.core_openai.execute_structured_chat_completion(system_prompt, raw_messages, response_format, tools=tools)