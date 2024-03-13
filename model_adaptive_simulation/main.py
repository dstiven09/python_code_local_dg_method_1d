import time
import os
from model_adaptive_test_case import ModelAdaptiveTestCase
from projection_criterion import ProjectionCriterion, ProjectionCriterionType
from simulation_result import SimulationResult
from smooth_bathymetry_model import SmoothBathymetryModel
from smooth_bathymetry_numerical_parameters import SmoothBathymetryNumericalParameters
from solitary_wave_model import SolitaryWaveModel
from solitary_wave_numerical_parameters import SolitaryWaveNumericalParameters
from beji_battjes_model import BejiBattjesModel
from beji_battjes_numerical_parameters import BejiBattjesNumericalParameters


def simulation_run():
    # Select a model and corresponding numerical paramters
    #model = SolitaryWaveModel()
    #numerical_parameters = SolitaryWaveNumericalParameters()

    model = BejiBattjesModel()
    numerical_parameters = BejiBattjesNumericalParameters()

    #model = SmoothBathymetryModel()
    #numerical_parameters = SmoothBathymetryNumericalParameters()

    # Setup the solver for the test case and select a projection criterion with a threshold
    test_case = ModelAdaptiveTestCase(model, numerical_parameters)
    criterion = ProjectionCriterion(ProjectionCriterionType.U_X_PROTECTED, 0.002)

    print(f'Starting criterion with {criterion.type}.')
    run_time = time.time()
    simulation_result = test_case.run(criterion)
    run_time = time.time() - run_time
    print(f'Time for one run: {run_time:.2f}s.')

    simulation_result.runtime = run_time
    #simulation_result.criteria = str(criterion.type)

    simulation_result.save('results/beji_battjes_u_x_protected_new.pkl')


def visualize_results():
    simulation_result = SimulationResult.load('results/beji_battjes_u_x_protected_new.pkl')
    #simulation_result.plot_criteria_norm()
    for i in [0,100,223, 300, 399]:
        #simulation_result.plot_water_hight_at_index(i)
        simulation_result.plot_solution_at_index(i)


if __name__ == "__main__":
    #simulation_run()
    visualize_results()
