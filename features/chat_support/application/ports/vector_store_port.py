from abc import ABC, abstractmethod
from typing import List
from ...domain.models import SupportContextChunk

class VectorStorePort(ABC):
    @abstractmethod
    def retrieve_relevant_chunks(self, query: str, limit: int = 2, collection_name: str | None = None) -> List[SupportContextChunk]:
        pass

    @abstractmethod
    def retrieve_keyword_chunks(self, query: str, limit: int = 2, collection_name: str | None = None) -> List[SupportContextChunk]:
        pass