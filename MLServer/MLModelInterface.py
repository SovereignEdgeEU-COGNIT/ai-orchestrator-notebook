from abc import ABC, abstractmethod

class MLModelInterface(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self, data) -> int:
        pass
