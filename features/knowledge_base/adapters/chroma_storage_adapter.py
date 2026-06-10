import json
from typing import Any, Dict, List, Optional

from core.infrastructure.chroma_client import CoreChromaClient

from ..application.ports.vector_management_port import VectorManagementPort
from ..application.ports.vector_storage_port import VectorStoragePort
from ..domain.models import ProcessedChunk, VectorDocument


class ChromaStorageAdapter(VectorStoragePort, VectorManagementPort):
    def __init__(self, core_chroma: CoreChromaClient):
        self.core_chroma = core_chroma
        try:
            self.col = self.core_chroma.collection
        except Exception:
            self.col = None

    def save_chunks(self, chunks: List[ProcessedChunk], collection_name: Optional[str] = None) -> None:
        ids = [chunk.id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        vectors = [chunk.embedding for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk.id,
                "document_name": chunk.metadata.get("document_name", chunk.source_file),
                "source_file": chunk.source_file,
                "file": chunk.source_file,
            }
            if collection_name:
                metadata["collection_name"] = collection_name
            metadatas.append(self._sanitize_metadata(metadata))

        collection = self._get_collection(collection_name)
        if not collection:
            return

        collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)
        print(f"[ChromaAdapter] Inserted {len(ids)} vectors into collection '{collection.name}'")

    def list_documents(self) -> List[VectorDocument]:
        return self.get_chunks()

    def get_document(self, doc_id: str, collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        chunks = self.get_chunks(collection_name=collection_name, chunk_id=doc_id)
        return chunks[0] if chunks else None

    def delete_document(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        return self.delete_chunks(collection_name=collection_name, chunk_id=doc_id) > 0

    def upsert_document(self, doc: VectorDocument, collection_name: Optional[str] = None) -> VectorDocument:
        collection = self._get_collection(collection_name)
        if not collection:
            return doc
        try:
            collection.delete(ids=[doc.id])
        except Exception:
            pass

        metadata = {
            **doc.metadata,
            "chunk_id": doc.id,
        }
        if "document_name" not in metadata:
            metadata["document_name"] = metadata.get("source_file", doc.id)
        if collection_name:
            metadata["collection_name"] = collection_name

        add_kwargs = {
            "ids": [doc.id],
            "documents": [doc.content],
            "metadatas": [self._sanitize_metadata(metadata)],
        }
        if doc.embedding:
            add_kwargs["embeddings"] = [doc.embedding]
        collection.add(**add_kwargs)
        return doc

    def list_collections(self) -> List[str]:
        try:
            cols = self.core_chroma.chroma_client.list_collections()
            names = []
            for collection in cols:
                name = getattr(collection, "name", None) or (
                    collection.get("name") if isinstance(collection, dict) else None
                )
                if name:
                    names.append(name)
            return names
        except Exception:
            return []

    def delete_collection(self, collection_name: str) -> bool:
        if not collection_name:
            return False
        try:
            self.core_chroma.chroma_client.delete_collection(name=collection_name)
            if self.col and getattr(self.col, "name", None) == collection_name:
                self.col = None
            return True
        except Exception:
            return False

    def get_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> List[VectorDocument]:
        collection = self._get_collection(collection_name)
        if not collection:
            return []

        try:
            get_kwargs: Dict[str, Any] = {"include": ["documents", "metadatas", "embeddings"]}
            if chunk_id:
                get_kwargs["ids"] = [chunk_id]
            if document_name:
                get_kwargs["where"] = {"document_name": document_name}

            data = collection.get(**get_kwargs)
            return self._to_vector_documents(data)
        except Exception:
            return []

    def delete_chunks(
        self,
        collection_name: Optional[str] = None,
        document_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> int:
        collection = self._get_collection(collection_name)
        if not collection:
            return 0

        chunks = self.get_chunks(
            collection_name=collection_name,
            document_name=document_name,
            chunk_id=chunk_id,
        )
        ids = [chunk.id for chunk in chunks]
        if not ids:
            return 0

        try:
            collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def _get_collection(self, collection_name: Optional[str] = None):
        if collection_name:
            return self.core_chroma.chroma_client.get_or_create_collection(name=collection_name)
        return self.col

    def _to_vector_documents(self, data: Dict[str, Any]) -> List[VectorDocument]:
        ids = self._flatten_chroma_value(data.get("ids"))
        docs = self._flatten_chroma_value(data.get("documents"))
        metas = self._flatten_chroma_value(data.get("metadatas"))
        embeddings = self._flatten_chroma_value(data.get("embeddings"))

        results: List[VectorDocument] = []
        for index, doc_id in enumerate(ids):
            content = self._value_at(docs, doc_id, index, "")
            metadata = self._value_at(metas, doc_id, index, {}) or {}
            embedding = self._normalize_embedding(self._value_at(embeddings, doc_id, index, None))
            results.append(
                VectorDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                )
            )
        return results

    def _flatten_chroma_value(self, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value

    def _value_at(self, values: Any, doc_id: str, index: int, default: Any) -> Any:
        if isinstance(values, dict):
            return values.get(doc_id, default)
        # Soporta tanto listas tradicionales de Python como arrays de numpy
        if not isinstance(values, (str, dict)) and hasattr(values, "__getitem__") and index < len(values):
            return values[index]
        return default

    def _normalize_embedding(self, embedding: Any) -> Optional[List[float]]:
        if embedding is None:
            return None
        try:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            return list(embedding)
        except Exception:
            try:
                return [float(embedding)]
            except Exception:
                return None

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
        sanitized: Dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
        return sanitized
