from abc import ABC, abstractmethod

class Query(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def query(self):
        pass