from typing import Optional
from pydantic import BaseModel
from ..ports.telemetry_port import TelemetryPort


class RecordUserFeedbackInput(BaseModel):
    trace_id: str
    value: int  # 1 para 👍, 0 o -1 para 👎
    comment: Optional[str] = None


class RecordUserFeedbackUseCase:
    """Caso de uso para registrar el feedback directo brindado por el usuario en la interfaz."""
    def __init__(self, telemetry: TelemetryPort):
        self.telemetry = telemetry

    def execute(self, input_data: RecordUserFeedbackInput) -> None:
        if not input_data.trace_id:
            raise ValueError("El trace_id es requerido para registrar feedback.")
            
        print(f"[Feedback] Registrando feedback del usuario para trace_id={input_data.trace_id} (valor={input_data.value})...")
        self.telemetry.record_evaluation_score(
            trace_id=input_data.trace_id,
            name="user-feedback",
            value=float(input_data.value),
            comment=input_data.comment
        )
