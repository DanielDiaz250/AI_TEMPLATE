from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from core.infrastructure.openai_client import CoreOpenAIClient
from core.infrastructure.chroma_client import CoreChromaClient

from .domain.models import DocumentInput
from .application.insert_use_cases import IngestDocumentUseCase, IngestDocumentOutput
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
from .adapters.openai_embedding_adapter import OpenAIEmbeddingAdapter
from .adapters.chroma_storage_adapter import ChromaStorageAdapter
from .application.list_use_case import ListDocsUseCase
from .application.get_use_case import GetDocUseCase
from .application.upsert_use_case import UpsertDocUseCase
from .application.delete_use_case import DeleteDocUseCase
from .domain.models import VectorDocument

kb_router = APIRouter(prefix="/knowledge-base", tags=["Knowledge Base Management"])

shared_openai_client = CoreOpenAIClient()
shared_chroma_client = CoreChromaClient()

embedding_adapter = OpenAIEmbeddingAdapter(core_client=shared_openai_client)
storage_adapter = ChromaStorageAdapter(core_chroma=shared_chroma_client)

ingest_use_case = IngestDocumentUseCase(
    embedding_service=embedding_adapter, vector_storage=storage_adapter
)

# management CRUD use cases
list_use_case = ListDocsUseCase(storage=storage_adapter)
get_use_case = GetDocUseCase(storage=storage_adapter)
upsert_use_case = UpsertDocUseCase(storage=storage_adapter)
delete_use_case = DeleteDocUseCase(storage=storage_adapter)


@kb_router.get("/management/v1/docs", status_code=status.HTTP_200_OK)
def list_docs(collection_name: str | None = Query(None)):
    if collection_name:
        return [d.dict() for d in get_use_case.get_chunks(collection_name=collection_name)]
    return [d.dict() for d in list_use_case.execute()]


@kb_router.get("/management/v1/docs/{doc_id}", status_code=status.HTTP_200_OK)
def get_doc(doc_id: str, collection_name: str | None = Query(None)):
    doc = get_use_case.execute(doc_id, collection_name=collection_name)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc.dict()


@kb_router.post("/management/v1/docs", status_code=status.HTTP_201_CREATED)
def upsert_doc(payload: VectorDocument, collection_name: str | None = Query(None)):
    saved = upsert_use_case.execute(payload, collection_name=collection_name)
    return saved.dict()


@kb_router.delete("/management/v1/docs/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_doc(doc_id: str, collection_name: str | None = Query(None)):
    ok = delete_use_case.execute(doc_id, collection_name=collection_name)
    if not ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Delete failed")
    return {}


@kb_router.get("/management/v1/collections", status_code=status.HTTP_200_OK)
def list_collections():
    return get_use_case.list_collections()


@kb_router.delete("/management/v1/collections/{collection_name}", status_code=status.HTTP_200_OK)
def delete_collection(collection_name: str):
    ok = delete_use_case.delete_collection(collection_name)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found or delete failed")
    return {"deleted": True, "collection_name": collection_name}


@kb_router.get("/management/v1/collections/{collection_name}/chunks", status_code=status.HTTP_200_OK)
def get_collection_chunks(
    collection_name: str,
    document_name: str | None = Query(None),
    chunk_id: str | None = Query(None),
):
    chunks = get_use_case.get_chunks(
        collection_name=collection_name,
        document_name=document_name,
        chunk_id=chunk_id,
    )
    return [c.dict() for c in chunks]


@kb_router.get("/management/v1/collections/{collection_name}/documents/{document_name}/chunks", status_code=status.HTTP_200_OK)
def get_document_chunks(collection_name: str, document_name: str):
    chunks = get_use_case.get_chunks(collection_name=collection_name, document_name=document_name)
    return [c.dict() for c in chunks]


@kb_router.delete("/management/v1/collections/{collection_name}/documents/{document_name}/chunks", status_code=status.HTTP_200_OK)
def delete_document_chunks(collection_name: str, document_name: str):
    deleted = delete_use_case.delete_chunks(collection_name=collection_name, document_name=document_name)
    return {"deleted": deleted, "collection_name": collection_name, "document_name": document_name}


@kb_router.get("/management/v1/collections/{collection_name}/chunks/{chunk_id}", status_code=status.HTTP_200_OK)
def get_collection_chunk(collection_name: str, chunk_id: str):
    chunks = get_use_case.get_chunks(collection_name=collection_name, chunk_id=chunk_id)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
    return chunks[0].dict()


@kb_router.delete("/management/v1/collections/{collection_name}/chunks", status_code=status.HTTP_200_OK)
def delete_collection_chunks(
    collection_name: str,
    document_name: str | None = Query(None),
    chunk_id: str | None = Query(None),
):
    if not document_name and not chunk_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide document_name or chunk_id to delete chunks",
        )
    deleted = delete_use_case.delete_chunks(
        collection_name=collection_name,
        document_name=document_name,
        chunk_id=chunk_id,
    )
    return {"deleted": deleted}

@kb_router.post("/v1/ingest", response_model=IngestDocumentOutput, status_code=status.HTTP_201_CREATED)
async def ingest_document_endpoint(
    payload: DocumentInput,
    collection_name: str | None = Query(None),
    strategy: str = Query("recursive"),
):
    try:
        return ingest_use_case.execute(payload, strategy=strategy, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


class BatchIn(BaseModel):
    title: str
    texts: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


@kb_router.post("/v1/ingest/batch", response_model=IngestDocumentOutput, status_code=status.HTTP_201_CREATED)
async def ingest_batch_endpoint(
    payload: BatchIn,
    collection_name: str | None = Query(None),
    strategy: str = Query("recursive"),
):
    try:
        raw = "\n\n".join(payload.texts)
        doc = DocumentInput(title=payload.title, raw_content=raw, metadata=payload.metadata)
        return ingest_use_case.execute(doc, strategy=strategy, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@kb_router.post("/v1/ingest/upload", response_model=IngestDocumentOutput, status_code=status.HTTP_201_CREATED)
async def ingest_upload_endpoint(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    metadata: str | None = Form(None),
    collection_name: str | None = Form(None),
    strategy: str = Form("recursive"),
):
    try:
        raw_bytes = await file.read()
        text = raw_bytes.decode('utf-8', errors='ignore')
        doc_title = title or getattr(file, 'filename', 'uploaded_document')
        meta_obj = {}
        if metadata:
            try:
                meta_obj = json.loads(metadata)
            except Exception:
                meta_obj = {"raw_metadata": metadata}

        doc = DocumentInput(title=doc_title, raw_content=text, metadata=meta_obj)
        return ingest_use_case.execute(doc, strategy=strategy, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    


