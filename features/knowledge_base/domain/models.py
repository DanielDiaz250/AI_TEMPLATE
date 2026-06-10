import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

class DocumentInput(BaseModel):
    title: str
    raw_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessedChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source_file: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class VectorDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_document_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" not in data and "document" in data:
            data = {**data, "content": data["document"]}
        return data

    @property
    def document(self) -> str:
        return self.content
