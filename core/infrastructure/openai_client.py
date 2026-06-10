import os
from typing import List, Dict, Any, Optional
import openai
from langfuse.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Intentar cargar variables desde un archivo .env si está presente.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Fallback simple: leer .env en workspace root si existe (no dependemos de python-dotenv)
    try:
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        env_path = os.path.normpath(env_path)
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k.strip(), v)
    except Exception:
        pass

class CoreOpenAIClient:
    """Cliente Core que usa OpenAI para embeddings y chat.

    Lee `OPENAI_API_KEY` del entorno y permite configurar `EMBEDDING_MODEL` y `CHAT_MODEL`.
    """
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("CRÍTICO: Faltan llaves en el archivo .env (Se requiere OPENAI_API_KEY).")

        # Asegurar compatibilidad de variables de host para Langfuse SDK
        base_url = os.getenv("LANGFUSE_BASE_URL")
        if base_url and not os.getenv("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = base_url

        # Cliente único apuntando a OpenAI con tracing automático de Langfuse
        self.client = OpenAI(api_key=openai_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6), reraise=True)
    def execute_embedding(self, text: str) -> List[float]:
        """Llamada real a OpenAI para vectorizar el texto del manual de IT."""
        print(f"[Core - OpenAI] Generando embedding...")
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except openai.NotFoundError as nf:
            raise RuntimeError(
                f"OpenAI embedding model not found: '{self.embedding_model}'. "
                "Set the EMBEDDING_MODEL env var to a valid model name for OpenAI. "
                f"Original error: {nf}"
            )
        except openai.OpenAIError as oe:
            raise RuntimeError(f"OpenAI embedding request failed: {oe}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6), reraise=True)
    def execute_chat_completion(
        self,
        system_prompt: str,
        conversation_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Llamada real a OpenAI para la respuesta del Chatbot de soporte."""
        print(f"[Core - OpenAI] Solicitando inferencia de chat de alta velocidad (tools={True if tools else False})...")
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(conversation_messages)

        try:
            kwargs = {
                "model": self.chat_model,
                "messages": formatted_messages,
                "temperature": 0.15
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            if message.tool_calls:
                return message
            return message.content
        except openai.NotFoundError as nf:
            raise RuntimeError(
                f"OpenAI chat model not found: '{self.chat_model}'. "
                "Set the CHAT_MODEL env var to a valid model name for OpenAI. "
                f"Original error: {nf}"
            )
        except openai.OpenAIError as oe:
            raise RuntimeError(f"OpenAI chat request failed: {oe}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6), reraise=True)
    def execute_structured_chat_completion(
        self,
        system_prompt: str,
        conversation_messages: List[Dict[str, Any]],
        response_format: Any,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Llamada real a OpenAI para obtener una respuesta estructurada validada con Pydantic."""
        print(f"[Core - OpenAI] Solicitando inferencia de chat estructurada (tools={True if tools else False})...")
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(conversation_messages)

        try:
            kwargs = {
                "model": self.chat_model,
                "messages": formatted_messages,
                "response_format": response_format,
                "temperature": 0.15
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.beta.chat.completions.parse(**kwargs)
            message = response.choices[0].message
            
            # Si el modelo decidió llamar a herramientas, retornamos el mensaje crudo con tool_calls
            if message.tool_calls:
                return message
                
            return message.parsed
        except openai.NotFoundError as nf:
            raise RuntimeError(
                f"OpenAI chat model not found: '{self.chat_model}'. "
                f"Original error: {nf}"
            )
        except openai.OpenAIError as oe:
            raise RuntimeError(f"OpenAI structured chat request failed: {oe}")