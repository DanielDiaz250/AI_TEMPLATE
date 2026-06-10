from ..application.ports.embedding_port import EmbeddingPort
from core.infrastructure.openai_client import CoreOpenAIClient

class OpenAIEmbeddingAdapter(EmbeddingPort):
    def __init__(self, core_client: CoreOpenAIClient):
        self.core_client = core_client

    def calculate_embedding(self, text: str) -> list[float]:
        return self.core_client.execute_embedding(text)