from abc import abstractmethod, ABC


class NumericalParameters(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass
