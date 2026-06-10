from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class TelemetryPort(ABC):
    """
    Puerto (interfaz abstracta) que define los servicios de observabilidad,
    trazabilidad y auditoría de la aplicación.
    Permite registrar eventos personalizados y almacenar puntuaciones de evaluación
    sin acoplar la lógica de negocio a un proveedor específico (ej. Langfuse, Datadog).
    """

    @abstractmethod
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Registra un evento personalizado en el colector de telemetría.

        Args:
            event_name (str): Nombre descriptivo del evento (ej. 'chat_started').
            payload (Dict[str, Any]): Datos estructurados adicionales asociados al evento.
        """
        pass

    @abstractmethod
    def record_evaluation_score(self, trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
        """
        Almacena una métrica de evaluación (Score) asociada a una traza de conversación.

        Args:
            trace_id (str): Identificador único de la traza de conversación en el sistema.
            name (str): Nombre de la métrica (ej. 'fidelidad-rag', 'user-feedback').
            value (float): Puntuación numérica asignada (ej. de 1.0 a 5.0).
            comment (Optional[str], optional): Comentario o justificación paso a paso (reasoning).
        """
        pass