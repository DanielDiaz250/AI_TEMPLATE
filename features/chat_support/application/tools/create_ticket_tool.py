SCHEMA = {
    "type": "function",
    "function": {
        "name": "crear_ticket_soporte",
        "description": "Escala formalmente el problema creando un ticket de soporte técnico cuando el asistente no puede resolver la duda o a petición explícita del usuario.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Descripción detallada del problema técnico, error o solicitud del usuario que se incluirá en el ticket."
                }
            },
            "required": ["description"],
            "additionalProperties": False
        }
    }
}
