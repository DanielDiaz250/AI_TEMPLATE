from typing import List
from features.knowledge_base.application.ports.vector_management_port import VectorManagementPort
from features.knowledge_base.domain.models import VectorDocument


class ListDocsUseCase:
    def __init__(self, storage: VectorManagementPort):
        self.storage = storage

    def execute(self) -> List[VectorDocument]:
        return self.storage.list_documents()
