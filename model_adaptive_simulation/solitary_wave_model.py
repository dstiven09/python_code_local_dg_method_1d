"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)

This test case is an analytical solution to the non-hydrostatic extension for the shallow water eqations with quadratic vertical pressure profile as described in Jeschke (2017) and Sebra-Santos (1987).

Jeschke, A., Pedersen, G.K., Vater, S., Behrens, J.: Depth-averaged non-hydrostatic extension for shallow water equations with quadratic vertical pressure profile: Equivalence to boussinesq-type equations. International Journal for Numerical Methods in Fluids (2017). DOI:10.1002/fld.4361. URL http://dx.doi.org/10.1002/fld.4361.

Seabra-Santos, F.J., Renuoard, D.P., Temperville, A.M.: Numerical and experimental study of the transformation of a solitary wave over a shelf or isolated obstacle. Journal of Fluid Mechanics 176, 117-134 (1987)

"""

from hypy1d.elliptic.equation import AnalyticalSolution
from hypy1d.grid import Grid1D
import numpy as np
from hypy1d.dg_element import DGElement1D
from model import Model


class SolitaryWaveModel(Model):

    def __init__(self) -> None:
        self.u_0 = 0.0
        self.A = 1.5  # This has to be > 0
        self.h_0 = 0.4
        self.still_water_depth = 10.0
        # Parameters for the boundary condition
        self.a = 0.01
        self.T = 2.02
        self.t_max = 20.0

        self.x_min = 0.0  # position of left boundary
        self.x_max = 3200.0  # position of right boundary
        self.number_of_unknowns = 4

        self.gravitational_constant = 9.80616
        self.is_non_linear_non_hydrostatic = 1
        self.swnl = 1
        self.B = 0  # variable f_d
        self.has_periodic_boundary = True

    def boundary_values(self, dg_element: DGElement1D, grid: Grid1D) -> list:
        return []

    def initial_condition(self, dg_element: DGElement1D, dt: float):
        t = 0.
        analytical_solution = AnalyticalSolution(dg_element,
                                                 self.gravitational_constant,
                                                 self.still_water_depth,
                                                 self.A, self.B)
        current_q = np.zeros((dg_element.doflength, self.number_of_unknowns))
        previous_q = np.zeros((dg_element.doflength, self.number_of_unknowns))
        bathymetry = np.zeros(dg_element.doflength)

        a = 1.
        x0 = (max(dg_element.dofcoordinates) -
              min(dg_element.dofcoordinates)) / 4.

        current_q = analytical_solution.analytical_solit(t, a, x0)
        previous_q = analytical_solution.analytical_solit(t - dt, a, x0)

        return current_q, previous_q, bathymetry
