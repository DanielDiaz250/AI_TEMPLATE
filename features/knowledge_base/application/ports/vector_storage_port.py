from abc import ABC, abstractmethod
from typing import List, Optional
from ...domain.models import ProcessedChunk


class VectorStoragePort(ABC):
    @abstractmethod
    def save_chunks(self, chunks: List[ProcessedChunk], collection_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def delete_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> int:
        pass
