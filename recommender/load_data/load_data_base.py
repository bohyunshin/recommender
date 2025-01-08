from abc import ABC, abstractmethod

class LoadDataBase(ABC):
    def __init__(self, **kwargs):
        """
        Abstract base class for all data loaders.
        """
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
        Abstract method for loading data.
        """
        raise NotImplementedError