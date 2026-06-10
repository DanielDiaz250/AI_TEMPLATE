import random
from ..application.ports.ticket_system_port import TicketSystemPort

class MockTicketAdapter(TicketSystemPort):
    def create_ticket(self, user_id: str, description: str) -> str:
        """Simula la creación de un ticket en un sistema de ticketing corporativo."""
        ticket_number = random.randint(10000, 99999)
        ticket_id = f"TICKET-VAL-{ticket_number}"
        print(f"[MockTicketAdapter] Ticket creado para el usuario '{user_id}': ID={ticket_id}, descripción='{description}'")
        return ticket_id
