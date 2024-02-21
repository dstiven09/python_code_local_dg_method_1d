from hypy1d.boundary_value import BoundaryValueDGInflowOutflow
from hypy1d.grid import Grid1D
import numpy as np
from hypy1d.dg_element import DGElement1D
from model import Model


class SmoothBathymetryModel(Model):

    def __init__(self) -> None:
        self.u_0 = 0.0
        self.A = 1.5
        self.h_0 = 0.4
        self.still_water_depth = self.h_0

        # Parameters for the boundary condition
        self.a = 0.01
        self.T = 2.02
        self.t_max = 40.0  # 40.0   # end time

        self.x_min = 0.0  # position of left boundary
        self.x_max = 40.0  # position of right boundary
        self.number_of_unknowns = 4

        self.gravitational_constant = 9.80616
        self.is_non_linear_non_hydrostatic = 1
        self.swnl = 1
        self.B = 0  # variable f_d
        self.has_periodic_boundary = False

    def boundary_values(self, dg_element: DGElement1D,
                            grid: Grid1D) -> list:
        return [
            BoundaryValueDGInflowOutflow(grid, grid.bnodes[0], dg_element,
                                         self.gravitational_constant, self.h_0,
                                         self.u_0, self.ssh_left,
                                         self.number_of_unknowns,
                                         self.trac_left),
            BoundaryValueDGInflowOutflow(grid, grid.bnodes[1], dg_element,
                                         self.gravitational_constant, self.h_0,
                                         self.u_0, self.ssh_right,
                                         self.number_of_unknowns,
                                         self.trac_right)
        ]

    def ssh_left(self, t):
        return self.a * np.sin(2. * np.pi * t / self.T)

    def ssh_right(self, t):
        return 0.0

    def trac_left(self, t):
        return self.h_0 * self.a * (0.5 -
                                    0.5 * np.cos(2. * np.pi * t / self.T))

    def trac_right(self, t):
        return self.h_0 * 0.0

    def initial_condition(self, dg_element: DGElement1D, dt: float):

        h_max = 0.3
        m1 = 0.05
        m2 = 0.1
        x1 = 6.
        x2 = x1 + h_max / m1
        x3 = x2 + 2.
        x4 = x3 + h_max / m2

        current_q = np.zeros((dg_element.doflength, self.number_of_unknowns))
        bathymetry = np.zeros(dg_element.doflength)
        hw = np.zeros(dg_element.doflength)
        hpnh = np.zeros(dg_element.doflength)

        bathymetry = h_max * np.exp(-np.power(dg_element.dofcoordinates -
                                              12, 2))

        h = self.still_water_depth - bathymetry
        hu = 0.

        hw = 0.
        hpnh = 0.
        current_q[:, 2] = hw
        current_q[:, 3] = hpnh

        current_q[:, 0] = h
        current_q[:, 1] = hu
        previous_q = np.copy(current_q)
        return current_q, previous_q, bathymetry
