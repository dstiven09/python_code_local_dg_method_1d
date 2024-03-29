import numpy as np

from hypy1d.grid import Grid1D, generate_uniformgrid
from hypy1d.dg_element import DGReferenceElement1D, DGElement1D, DGQuadrature1D
import hypy1d.swe.riemann_solvers as riemann_solvers
from hypy1d.swe.equation import EqSWESource
from hypy1d.timestepping_r2 import RungeKuttaDG
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
from hypy1d.limiter_none import Limiter

from hypy1d.elliptic.equation import EllipticEquation, Localnh
from factors_local_model_adaptive import FactorsElliptic, EllipticSolver
from hypy1d.boundary_elliptic_local import LeftDirichlet
from hypy1d.boundary_elliptic_local import RightDirichlet
from numerical_parameters import NumericalParameters

from projection_criterion import ProjectionCriterion, ProjectionCriterionType
from simulation_result import SimulationResult
from model import Model


class ModelAdaptiveTestCase:

    def __init__(self, model: Model,
                 numerical_parameters: NumericalParameters) -> None:
        self.model = model
        self.numerical_parameters = numerical_parameters

    def run(self,
            projection_criterion: ProjectionCriterion) -> SimulationResult:

        reference_element = DGReferenceElement1D(
            self.numerical_parameters.interpolation_order)
        node_coordinates, element_nodes, neighboring_elements = generate_uniformgrid(
            self.model.x_min,
            self.model.x_max,
            self.numerical_parameters.number_of_elements,
            periodic=self.model.has_periodic_boundary)
        grid = Grid1D(node_coordinates, element_nodes, neighboring_elements)
        dg_element = DGElement1D(grid, reference_element)
        dg_quadratur = DGQuadrature1D(
            reference_element, self.numerical_parameters.interpolation_order)

        current_q, previous_q, bathymetry = self.model.initial_condition(
            dg_element, self.numerical_parameters.dt)

        local_nh = Localnh(grid, dg_element)
        shallow_water_equation = riemann_solvers.EqSWERusanov(
            local_nh, self.model.still_water_depth,
            self.model.gravitational_constant, self.model.swnl,
            self.numerical_parameters.discretization_order, self.model.A,
            self.model.B, 1.0e-8, self.model.number_of_unknowns - 2)
        source = EqSWESource(shallow_water_equation, dg_element, dg_quadratur)

        boundary_values = self.model.boundary_values(dg_element, grid)
        flux_divergence = FluxDivergence1DDG(grid, shallow_water_equation,
                                             source, dg_element, dg_quadratur,
                                             boundary_values)
        limiter = Limiter(grid, shallow_water_equation, source, dg_element)
        runge_kutta = RungeKuttaDG(flux_divergence, limiter, 2)
        source.update(bathymetry)
        elliptic_equation = EllipticEquation(
            dg_element, self.model.gravitational_constant,
            self.model.still_water_depth, bathymetry, self.model.A,
            self.model.B, self.model.is_non_linear_non_hydrostatic,
            self.numerical_parameters.discretization_order)
        elliptic_factors = FactorsElliptic(grid, source, elliptic_equation,
                                           dg_element, dg_quadratur, local_nh)
        elliptic_solver = EllipticSolver()
        elliptic_factors.update_local_projection_elements(
            flux_divergence, current_q, current_q, dg_element, 0.,
            projection_criterion)

        dirichlet_boundary = np.array([])
        boundary_element = [
            LeftDirichlet(grid, dg_element, local_nh),
            RightDirichlet(grid, dg_element, local_nh), dirichlet_boundary
        ]

        step_index = 0
        t = 0.0
        simulation_result = SimulationResult(grid.nodecoordinates,
                                             dg_element.elementdofs,
                                             dg_element.ddx, bathymetry,
                                             self.model.still_water_depth,
                                             projection_criterion,
                                             time_step = self.numerical_parameters.dt)
        while t < self.model.t_max - np.finfo(float).resolution:
            if step_index == self.numerical_parameters.max_iterations:
                break
            u = np.zeros(dg_element.doflength)
            CFLg = np.max(u +
                          np.sqrt(shallow_water_equation.g * current_q[:, 0])
                          ) * self.numerical_parameters.dt / np.min(
                              grid.elementwidth)

            next_q, q_help_local = runge_kutta.step(
                current_q, previous_q, t, self.numerical_parameters.dt)

            if step_index < self.numerical_parameters.number_of_global_steps:
                current_criterion = ProjectionCriterion(
                    ProjectionCriterionType.GLOBAL, 0.0)
            else:
                current_criterion = projection_criterion

            elliptic_factors.update_local_projection_elements(
                flux_divergence, next_q, current_q, dg_element, t,
                current_criterion)

            if (elliptic_factors.Local.doflenloc > 0):
                hu_local, dipnh_local = elliptic_solver.solve_system(
                    elliptic_factors, next_q, self.numerical_parameters.dt, t,
                    boundary_element, q_help_local)
                next_q = elliptic_solver.correct_tracer(
                    elliptic_factors, flux_divergence, next_q, hu_local,
                    dipnh_local, t, self.numerical_parameters.dt, q_help_local)
                # application of limiter because correction step for water height
                next_q = limiter(next_q)

            simulation_result.append_time_step(
                len(elliptic_factors.Local.dofloc), current_q,
                elliptic_factors.Local.dofloc)
            previous_q = current_q
            current_q = next_q

            t = t + self.numerical_parameters.dt
            step_index = step_index + 1
            print(f'Step index: {step_index}')

        if (np.isnan(CFLg)):
            quit()

        return simulation_result
