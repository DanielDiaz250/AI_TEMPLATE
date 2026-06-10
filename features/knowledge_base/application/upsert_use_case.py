from typing import Optional

from features.knowledge_base.application.ports.vector_management_port import VectorManagementPort
from features.knowledge_base.domain.models import VectorDocument


class UpsertDocUseCase:
    def __init__(self, storage: VectorManagementPort):
        self.storage = storage

    def execute(self, doc: VectorDocument, collection_name: Optional[str] = None) -> VectorDocument:
        return self.storage.upsert_document(doc, collection_name=collection_name)
