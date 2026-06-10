from typing import List, Optional
from features.knowledge_base.application.ports.vector_management_port import VectorManagementPort
from features.knowledge_base.domain.models import VectorDocument


class GetDocUseCase:
    def __init__(self, storage: VectorManagementPort):
        self.storage = storage

    def execute(self, doc_id: str, collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        return self.storage.get_document(doc_id, collection_name=collection_name)

    def list_collections(self) -> List[str]:
        return self.storage.list_collections()

    def get_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> List[VectorDocument]:
        return self.storage.get_chunks(
            collection_name=collection_name,
            document_name=document_name,
            chunk_id=chunk_id,
        )
