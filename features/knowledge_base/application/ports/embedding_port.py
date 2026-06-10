from abc import ABC, abstractmethod
from typing import List

class EmbeddingPort(ABC):
    @abstractmethod
    def calculate_embedding(self, text: str) -> List[float]:
        pass