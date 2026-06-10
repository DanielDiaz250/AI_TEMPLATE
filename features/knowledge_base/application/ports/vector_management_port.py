from abc import ABC, abstractmethod
from typing import List, Optional
from features.knowledge_base.domain.models import VectorDocument


class VectorManagementPort(ABC):
    @abstractmethod
    def list_documents(self) -> List[VectorDocument]:
        pass

    @abstractmethod
    def get_document(self, doc_id: str, collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def upsert_document(self, doc: VectorDocument, collection_name: Optional[str] = None) -> VectorDocument:
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def get_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> List[VectorDocument]:
        pass

    @abstractmethod
    def delete_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> int:
        pass
