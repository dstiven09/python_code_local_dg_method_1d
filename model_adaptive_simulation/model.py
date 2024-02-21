from abc import ABC, abstractmethod
from hypy1d.grid import Grid1D
from hypy1d.dg_element import DGElement1D


class Model(ABC):
    
    @abstractmethod
    def boundary_values(self, dg_element: DGElement1D,
                            grid: Grid1D) -> list:
        pass

    @abstractmethod
    def initial_condition(self, dg_element: DGElement1D, dt: float):
        pass
