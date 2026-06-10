from typing import Optional

from features.knowledge_base.application.ports.vector_management_port import VectorManagementPort


class DeleteDocUseCase:
    def __init__(self, storage: VectorManagementPort):
        self.storage = storage

    def execute(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        return self.storage.delete_document(doc_id, collection_name=collection_name)

    def delete_collection(self, collection_name: str) -> bool:
        return self.storage.delete_collection(collection_name)

    def delete_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> int:
        return self.storage.delete_chunks(
            collection_name=collection_name,
            document_name=document_name,
            chunk_id=chunk_id,
        )
