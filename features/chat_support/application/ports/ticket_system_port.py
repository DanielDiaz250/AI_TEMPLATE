from abc import ABC, abstractmethod

class TicketSystemPort(ABC):
    @abstractmethod
    def create_ticket(self, user_id: str, description: str) -> str:
        """Crea un ticket en el sistema externo y devuelve su ID único."""
        pass
