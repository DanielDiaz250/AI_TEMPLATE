from pydantic import BaseModel, Field
from typing import Optional
import re
import numpy as np
from .ports.embedding_port import EmbeddingPort
from .ports.vector_storage_port import VectorStoragePort
from ..domain.models import DocumentInput, ProcessedChunk

class IngestDocumentOutput(BaseModel):
    success: bool
    chunks_processed: int
    document_title: str
    collection_name: Optional[str] = None
    chunk_ids: list[str] = Field(default_factory=list)

class IngestDocumentUseCase:
    """
    Caso de Uso Empresarial: Orquesta el pipeline de segmentación, vectorización
    e indexación jerárquica o plana de manuales técnicos corporativos.
    """
    def __init__(self, embedding_service: EmbeddingPort, vector_storage: VectorStoragePort):
        self.embedding_service = embedding_service
        self.vector_storage = vector_storage

    def _fixed_size_chunker(self, text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
        """
        ESTRATEGIA 1: Fixed-Size Chunking (Segmentación de Tamaño Fijo).
        
        VENTAJAS:
            - Simplicidad computacional y predictibilidad absoluta del tamaño del string.
            - Rapidez de ejecución óptima a nivel de CPU (Complejidad lineal O(N)).
            - El overlap mitiga parcialmente la pérdida de tokens clave en las fronteras del corte.
        DESVENTAJAS EN PRODUCCIÓN:
            - Carece de conciencia semántica: rompe oraciones, párrafos e instrucciones de código 
              técnico (ej. scripts de Bash o trazas de Java) de forma matemática cruda.
            - Rompe la pureza matemática del embedding al mezclar temas dispares en el mismo bloque.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def _recursive_character_chunker(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """
        ESTRATEGIA 2: Recursive Character Chunking (Segmentación Recursiva por Caracteres).
        
        VENTAJAS:
            - Preserva la cohesión semántica natural de la documentación técnica.
            - Utiliza una jerarquía ordenada de separadores humanos: Párrafos ("\n\n"), 
              líneas ("\n"), espacios (" ") y finalmente caracteres vacíos ("").
            - Mantiene las instrucciones de soporte unidas en bloques contextuales homogéneos.
        DESVENTAJAS EN PRODUCCIÓN:
            - Bloques de código formateados o dumps de logs excesivamente largos obligarán al 
              algoritmo a cortar de forma rígida si superan el 'chunk_size' máximo permitido.
        """
        separators = ["\n\n", "\n", " ", ""]
        
        def _split_text(current_text: str, current_seps: list[str]) -> list[str]:
            if len(current_text) <= chunk_size or not current_seps:
                return [current_text] if current_text.strip() else []
                
            sep = current_seps[0]
            raw_splits = current_text.split(sep) if sep != "" else list(current_text)
                
            final_splits = []
            current_chunk = ""
            
            for part in raw_splits:
                join_sep = sep if current_chunk else ""
                potential_chunk = current_chunk + join_sep + part
                
                if len(potential_chunk) <= chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk.strip():
                        final_splits.append(current_chunk)
                    if len(part) > chunk_size:
                        final_splits.extend(_split_text(part, current_seps[1:]))
                        current_chunk = ""
                    else:
                        current_chunk = part
                        
            if current_chunk.strip():
                final_splits.append(current_chunk)
            return final_splits

        return _split_text(text, separators)

    def _parent_child_chunker(self, text: str, parent_size: int = 800, child_size: int = 150) -> list[dict]:
        """
        ESTRATEGIA 3: Parent-Child / Hierarchical Chunking (Segmentación Jerárquica).
        
        VENTAJAS:
            - Resuelve de forma definitiva la paradoja del tamaño del fragmento en RAG Enterprise.
            - Los fragmentos pequeños (Children) garantizan una alta densidad semántica y léxica 
              durante la fase de búsqueda matemática en la base vectorial.
            - Los fragmentos grandes (Parents) protegen el contexto completo, evitando que el LLM 
              alucine al recibir datos aislados o fragmentados.
        DESVENTAJAS EN PRODUCCIÓN:
            - Añade sobrecarga operacional en la base de datos al requerir mapeos relacionales.
        """
        # 1. Creamos fragmentos grandes (Parents) usando el algoritmo semántico recursivo
        raw_parents = self._recursive_character_chunker(text, chunk_size=parent_size)
        
        hierarchical_payloads = []
        for p_idx, parent_text in enumerate(raw_parents):
            parent_id = f"parent-node-{p_idx}"
            
            # 2. Segmentamos el contenido del padre en trozos pequeños de alta precisión (Children)
            raw_children = self._fixed_size_chunker(parent_text, chunk_size=child_size, overlap=20)
            
            for c_idx, child_text in enumerate(raw_children):
                hierarchical_payloads.append({
                    "child_id": f"{parent_id}-child-{c_idx}",
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "child_text": child_text
                })
        return hierarchical_payloads
    
    def _semantic_chunker(self, text: str, similarity_threshold: float = 0.82) -> list[str]:
        """
        ESTRATEGIA 4: Semantic Chunking (Basado en Similitud Matemática de Embeddings).
        
        VENTAJAS: Garantiza que cada trozo hable de exactamente un solo tema puro.
        DESVENTAJAS: Costo en tiempo de cómputo y API muy elevado durante la ingesta.
        """
        print("[Algoritmo - Semantic] Segmentando por distancia matemática...")
        # 1. Separación inicial por oraciones usando expresiones regulares simples
        sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return sentences

        # 2. Calcular los embeddings de todas las oraciones de forma secuencial mediante el Puerto
        embeddings = [self.embedding_service.calculate_embedding(s) for s in sentences]
        
        chunks = []
        current_chunk = sentences[0]
        
        # 3. Medir similitud coseno entre oraciones adyacentes para detectar rupturas de tema
        for i in range(len(sentences) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i+1])
            
            # Fórmula de Similitud Coseno
            cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            if cosine_similarity >= similarity_threshold:
                # Si la similitud es alta, pertenecen al mismo tema; se fusionan
                current_chunk += " " + sentences[i+1]
            else:
                # Ruptura semántica detectada: cerramos chunk actual y abrimos uno nuevo
                chunks.append(current_chunk)
                current_chunk = sentences[i+1]
                
        chunks.append(current_chunk)
        return chunks

    def _structural_layout_chunker(self, text: str) -> list[str]:
        """
        ESTRATEGIA 5: Structural / Layout Chunking (Simulado para Texto Plano/Markdown).
        
        VENTAJAS: Respeta tablas, títulos y bloques de código sin romper su semántica visual.
        DESVENTAJAS: Requiere que la data de entrada contenga marcas estructurales claras (como Markdown).
        """
        print("[Algoritmo - Structural] Segmentando preservando etiquetas de Markdown...")
        # En producción real aquí usarías librerías como LlamaParse o Unstructured.
        # Simulamos la segmentación buscando bloques de código (```), tablas (|) o títulos (#).
        chunks = []
        lines = text.split("\n")
        current_block = ""
        in_code_block = False
        
        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                current_block += line + "\n"
                if not in_code_block: # Al cerrar el bloque de código, consolidamos el chunk
                    chunks.append(current_block.strip())
                    current_block = ""
                continue
                
            if in_code_block:
                current_block += line + "\n"
                continue
                
            # Si encontramos un título principal del documento, cerramos el bloque previo
            if line.startswith("#") or line.startswith("==="):
                if current_block.strip():
                    chunks.append(current_block.strip())
                current_block = line + "\n"
            else:
                current_block += line + "\n"
                
        if current_block.strip():
            chunks.append(current_block.strip())
        return chunks

    def _agentic_chunker(self, text: str) -> list[str]:
        """
        ESTRATEGIA 6: Agentic Chunking (Segmentación Inteligente Dirigida por un LLM).
        
        VENTAJAS: Máxima comprensión humana y lógica de negocio contextual sobre los límites del texto.
        DESVENTAJAS: Latencia extrema. Introduce costos de tokens adicionales en la ingesta.
        """
        print("[Algoritmo - Agentic] Solicitando segmentación inteligente...")
        # En producción real, harías una llamada limpia a un LLM (ej. GPT-4o-mini) pasándole el texto 
        # con un prompt pidiendo que actúe como un Ingeniero de Datos y devuelva un JSON estructurado.
        # Simulamos el resultado aplicando un corte basado en los bloques principales de intención.
        sections = text.split("PÁRRAFO")
        return [f"PÁRRAFO {s.strip()}" for s in sections if s.strip()]

    def _build_chunk(self, document: DocumentInput, text_segment: str, index: int) -> ProcessedChunk:
        vector = self.embedding_service.calculate_embedding(text_segment)
        metadata = {
            **document.metadata,
            "document_name": document.title,
            "source_file": document.title,
            "chunk_index": index,
        }
        chunk = ProcessedChunk(
            content=text_segment,
            source_file=document.title,
            metadata=metadata,
            embedding=vector,
        )
        chunk.metadata["chunk_id"] = chunk.id
        return chunk

    def execute(self, document: DocumentInput, strategy: str = "recursive", collection_name: Optional[str] = None) -> IngestDocumentOutput:
        """
        Punto de entrada del caso de uso. Permite alternar dinámicamente la estrategia de 
        procesamiento de datos según las necesidades del documento corporativo.
        """
        # Eliminar chunks previos del mismo documento en la colección destino para evitar duplicados
        self.vector_storage.delete_chunks(collection_name=collection_name, document_name=document.title)

        processed_chunks = []

        if strategy == "fixed":
            raw_chunks = self._fixed_size_chunker(document.raw_content)
            for index, text_segment in enumerate(raw_chunks):
                processed_chunks.append(self._build_chunk(document, text_segment, index))
            self.vector_storage.save_chunks(processed_chunks, collection_name=collection_name)

        elif strategy == "recursive":
            raw_chunks = self._recursive_character_chunker(document.raw_content)
            for index, text_segment in enumerate(raw_chunks):
                processed_chunks.append(self._build_chunk(document, text_segment, index))
            self.vector_storage.save_chunks(processed_chunks, collection_name=collection_name)

        elif strategy == "parent_child":
            # Nota: Esta estrategia avanzada requiere que tu VectorStoragePort implemente
            # un método especializado para indexar la estructura jerárquica
            hierarchy = self._parent_child_chunker(document.raw_content)
            for index, node in enumerate(hierarchy):
                # El embedding se calcula únicamente sobre la densidad semántica del hijo pequeño
                chunk_entity = self._build_chunk(document, node["child_text"], index)
                chunk_entity.metadata.update(
                    {
                        "parent_id": node["parent_id"],
                        "parent_text": node["parent_text"],
                    }
                )
                # Inyectamos el ID relacional y el texto completo del padre en los metadatos de la entidad
                # Puedes expandir los metadatos de tu modelo si deseas guardar el parent_text real en disco
                processed_chunks.append(chunk_entity)
            
            self.vector_storage.save_chunks(processed_chunks, collection_name=collection_name)

        else:
            raise ValueError(f"Estrategia de chunking desconocida: {strategy}")

        return IngestDocumentOutput(
            success=True, 
            chunks_processed=len(processed_chunks), 
            document_title=document.title,
            collection_name=collection_name,
            chunk_ids=[chunk.id for chunk in processed_chunks],
        )
