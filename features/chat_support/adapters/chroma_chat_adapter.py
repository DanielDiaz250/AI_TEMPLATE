from typing import List
from ..application.ports.vector_store_port import VectorStorePort
from ..domain.models import SupportContextChunk
from core.infrastructure.chroma_client import CoreChromaClient
from core.infrastructure.openai_client import CoreOpenAIClient

class ChromaChatAdapter(VectorStorePort):
    """Adaptador real de lectura que conecta la pregunta del usuario con la persistencia en disco."""
    def __init__(self, core_chroma_client: CoreChromaClient, core_openai_client: CoreOpenAIClient):
        self.core_chroma = core_chroma_client
        self.core_openai = core_openai_client # Añadimos el cliente para calcular embeddings en caliente

    def retrieve_relevant_chunks(self, query: str, limit: int = 2, collection_name: str | None = None) -> List[SupportContextChunk]:
        print(f"[Adapter - Chat Storage] Generando vector para la consulta: '{query}'")
        
        # 1. Transformamos la pregunta real del usuario en un vector numérico real
        query_vector = self.core_openai.execute_embedding(query)

        # 2. Consultamos la base de datos de ChromaDB real en disco
        raw_results = self.core_chroma.query_similarity(query_embedding=query_vector, limit=limit, collection_name=collection_name)

        # 3. Mapeamos al dominio
        domain_chunks = []
        for doc in raw_results:
            chunk = SupportContextChunk(
                id=doc["id"],
                content=doc["document"],
                source_metadata=doc["metadata"]
            )
            domain_chunks.append(chunk)
            
        return domain_chunks

    def retrieve_keyword_chunks(self, query: str, limit: int = 2, collection_name: str | None = None) -> List[SupportContextChunk]:
        print(f"[Adapter - Chat Storage] Ejecutando búsqueda por palabra clave para: '{query}'")
        
        target_collection = self.core_chroma.collection
        if collection_name:
            try:
                target_collection = self.core_chroma.chroma_client.get_collection(name=collection_name)
            except Exception:
                target_collection = self.core_chroma.collection
                
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        if not keywords:
            keywords = [query.lower()]
            
        all_results = []
        for kw in keywords[:3]:
            try:
                data = target_collection.get(
                    where_document={"$contains": kw},
                    limit=limit
                )
                if data and data["ids"]:
                    for i in range(len(data["ids"])):
                        chunk = SupportContextChunk(
                            id=data["ids"][i],
                            content=data["documents"][i],
                            source_metadata=data["metadatas"][i] if data["metadatas"] else {}
                        )
                        all_results.append(chunk)
            except Exception as e:
                print(f"[Adapter - Chat Storage] Error en keyword search contains '{kw}': {e}")
                
        seen = set()
        deduplicated = []
        for c in all_results:
            if c.id not in seen:
                seen.add(c.id)
                deduplicated.append(c)
                
        return deduplicated[:limit]