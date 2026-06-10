from typing import Optional
from pydantic import BaseModel
from ..ports.telemetry_port import TelemetryPort


class RecordUserFeedbackInput(BaseModel):
    trace_id: str
    value: int  # 1 para 👍, 0 o -1 para 👎
    comment: Optional[str] = None


class RecordUserFeedbackUseCase:
    """
    Caso de uso para registrar el feedback directo brindado por el usuario en la interfaz.
    Recibe la calificación manual y la almacena en el colector de telemetría asociado a la traza.
    """
    def __init__(self, telemetry: TelemetryPort):
        """
        Inicializa el caso de uso inyectando el puerto de telemetría.

        Args:
            telemetry (TelemetryPort): Adaptador para almacenar el score de feedback.
        """
        self.telemetry = telemetry

    def execute(self, input_data: RecordUserFeedbackInput) -> None:
        """
        Ejecuta el registro de feedback del usuario en la traza.

        Args:
            input_data (RecordUserFeedbackInput): Datos conteniendo trace_id, calificación (1 o 0) y comentario.
        """
        if not input_data.trace_id:
            raise ValueError("El trace_id es requerido para registrar feedback.")
            
        print(f"[Feedback] Registrando feedback del usuario para trace_id={input_data.trace_id} (valor={input_data.value})...")
        self.telemetry.record_evaluation_score(
            trace_id=input_data.trace_id,
            name="user-feedback",
            value=float(input_data.value),
            comment=input_data.comment
        )
