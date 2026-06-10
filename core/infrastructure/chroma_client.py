import chromadb
from typing import List, Dict, Any

class CoreChromaClient:
    """Cliente real de persistencia en disco de ChromaDB para almacenamiento de vectores."""
    def __init__(self):
        # Establecemos persistencia física en el directorio local del proyecto
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Obtenemos o creamos la colección empresarial dedicada a soporte técnico de IT
        self.collection = self.chroma_client.get_or_create_collection(
            name="mas_global_it_knowledge"
        )
        print("[Core - ChromaDB] Conexión persistente en disco a './chroma_db' establecida con éxito.")

    def insert_vectors(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Inserta físicamente los documentos y sus vectores en el almacenamiento local."""
        print(f"[Core - ChromaDB] Insertando {len(ids)} vectores de forma persistente...")
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("[Core - ChromaDB] Indexación completada con éxito.")

    def query_similarity(self, query_embedding: List[float], limit: int = 2, collection_name: str | None = None) -> List[Dict[str, Any]]:
        """Realiza una consulta matemática real de distancia coseno en la base de datos."""
        print(f"[Core - ChromaDB] Ejecutando Query de similitud semántica en disco (colección: {collection_name or 'default'})...")
        
        target_collection = self.collection
        if collection_name:
            try:
                target_collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                print(f"[Core - ChromaDB] Colección '{collection_name}' no encontrada. Usando colección por defecto.")
                target_collection = self.collection

        results = target_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        # Mapeamos la respuesta del formato nativo de Chroma hacia una lista estándar de diccionarios
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                record = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                }
                formatted_results.append(record)
                
        return formatted_results