from hypy1d.boundary_value import BoundaryValueDGZeroExtrap
from hypy1d.elliptic.equation import AnalyticalSolution
from hypy1d.grid import Grid1D
import numpy as np
from hypy1d.dg_element import DGElement1D
from model import Model


class SolitaryWaveInundatedBeachModel(Model):

    def __init__(self) -> None:
        self.u_0 = 0.0
        self.A = 1.5  # This has to be > 0
        self.h_0 = 0.4
        self.still_water_depth = 1.0
        self.t_max = 40.0

        self.x_min = -30.0  # position of left boundary
        self.x_max = 70.0  # position of right boundary
        self.number_of_unknowns = 4

        self.gravitational_constant = 9.80616
        self.is_non_linear_non_hydrostatic = 1
        self.swnl = 1
        self.B = 0.0  # variable f_d
        self.has_periodic_boundary = False

    def boundary_values(self, dg_element: DGElement1D,
                            grid: Grid1D) -> list:
        return [
            BoundaryValueDGZeroExtrap(grid, grid.bnodes[0], dg_element),
            BoundaryValueDGZeroExtrap(grid, grid.bnodes[1], dg_element)
        ]

    def initial_condition(self, dg_element: DGElement1D, dt: float):
        """
        Definition of initial conditions: experiment of beji and battjes
        """
        t = 0.0
        analytical_solution = AnalyticalSolution(dg_element,
                                                 self.gravitational_constant,
                                                 self.still_water_depth,
                                                 self.A, self.B)
        a = 0.019 * self.still_water_depth
        c = -np.sqrt(self.gravitational_constant *
                    (self.still_water_depth + a))  # as for swe-run
        #these are the KdV-parameters:
        K = np.sqrt(3. * a /
                    (4. * self.still_water_depth * self.still_water_depth *
                     (self.still_water_depth)))

        beta2 = 19.85
        X0 = self.still_water_depth * beta2
        L = np.arccosh(np.sqrt(20.)) / (K * self.still_water_depth)
        x0 = X0 + L
        m = -1. / beta2

        bathymetry = m * (dg_element.dofcoordinates - X0)
        bathymetry[dg_element.dofcoordinates >= X0] = 0.0

        current_q = np.zeros((dg_element.doflength, self.number_of_unknowns))
        previous_q = np.zeros((dg_element.doflength, self.number_of_unknowns))
        current_q[:,
                  0], current_q[:,
                                1], current_q[:,
                                              2], current_q[:,
                                                            3] = analytical_solution.initial_simplebeach(
                                                                t, a, c,
                                                                bathymetry, x0)
        previous_q[:,
                   0], previous_q[:,
                                  1], previous_q[:,
                                                 2], previous_q[:,
                                                                3] = analytical_solution.initial_simplebeach(
                                                                    t - dt, a,
                                                                    c,
                                                                    bathymetry,
                                                                    x0)
        for i in range(dg_element.doflength):
            if (current_q[i,0] < 1e-8):
                current_q[i,0] = 0.
                current_q[i,1] = 0.
            if (previous_q[i, 0] < 1e-8):
                previous_q[i, 0] = 0.
                previous_q[i, 1] = 0.

        return current_q, previous_q, bathymetry
