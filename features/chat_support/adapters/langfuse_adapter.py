from typing import Dict, Any, Optional
from ..application.ports.telemetry_port import TelemetryPort

class LangfuseTelemetryAdapter(TelemetryPort):
    """
    Adaptador concreto para la telemetría del chatbot que implementa TelemetryPort.
    Utiliza el SDK oficial de Langfuse para enviar trazas, eventos y puntuaciones
    de evaluación (scores) directamente a Langfuse Cloud.
    """

    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Registra un evento simple imprimiéndolo en consola. 
        En producción, esto puede ser extendido para enviar logs a CloudWatch/Datadog.
        """
        print(f"[Observabilidad - Langfuse] EVENTO: '{event_name}' | METADATA: {payload}")

    def record_evaluation_score(self, trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
        """
        Registra una puntuación numérica (fidelidad, relevancia o feedback manual) 
        asociada a una traza en Langfuse Cloud usando el cliente oficial.
        """
        from langfuse import get_client
        try:
            client = get_client()
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
            client.flush()
            print(f"[Observabilidad - Langfuse] Score registrado: {name}={value} para trace_id={trace_id}")
        except Exception as e:
            print(f"[Observabilidad - Langfuse - Error] No se pudo guardar el score {name}: {e}")