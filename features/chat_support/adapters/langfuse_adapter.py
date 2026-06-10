from typing import Dict, Any, Optional
from ..application.ports.telemetry_port import TelemetryPort

class LangfuseTelemetryAdapter(TelemetryPort):
    def log_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        print(f"[Observabilidad - Langfuse] EVENTO: '{event_name}' | METADATA: {payload}")

    def record_evaluation_score(self, trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
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