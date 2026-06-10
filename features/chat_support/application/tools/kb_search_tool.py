SCHEMA = {
    "type": "function",
    "function": {
        "name": "buscar_en_base_de_conocimiento",
        "description": "Busca fragmentos de documentación técnica de IT y la plataforma BEAN relevantes para resolver la consulta del usuario.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta técnica, pregunta o término de búsqueda optimizado para buscar en la base de datos de soporte."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}
