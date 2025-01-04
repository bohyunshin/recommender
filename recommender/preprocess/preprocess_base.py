from abc import ABC, abstractmethod

class PreoprocessorBase(ABC):
    def __init__(self, **kwargs):
        """
        Abstract base class for all preprocessors.
        """
        pass

    @abstractmethod
    def preprocess(self, **kwargs):
        """
        Abstract method for preprocssing data.
        """
        raise NotImplementedError