import time
import os
import imageio
from multiprocessing import Pool
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
    model = SolitaryWaveModel()
    numerical_parameters = SolitaryWaveNumericalParameters()

    #model = BejiBattjesModel()
    #numerical_parameters = BejiBattjesNumericalParameters()

    #model = SmoothBathymetryModel()
    #numerical_parameters = SmoothBathymetryNumericalParameters()

    # Setup the solver for the test case and select a projection criterion with a threshold
    test_case = ModelAdaptiveTestCase(model, numerical_parameters)
    criterion = ProjectionCriterion(ProjectionCriterionType.U_X_PROTECTED, 0.00015)

    print(f'Starting criterion with {criterion.type}.')
    run_time = time.time()
    simulation_result = test_case.run(criterion)
    run_time = time.time() - run_time
    print(f'Time for one run: {run_time:.2f}s.')

    simulation_result.runtime = run_time

    #naming results - regularization
    if isinstance(model, BejiBattjesModel):
        model_name = 'beji_battjes'
    elif isinstance(model, SolitaryWaveModel):
        model_name = 'solitary_wave'
    elif isinstance(model, SmoothBathymetryModel):
        model_name = 'smooth_bathymetry'

    #Asking whether the criterion is Global
    if criterion.type.value == 1:
        filename = f'results/{model_name}_{criterion.type}_t={model.t_max}_dt={numerical_parameters.dt}_elements={numerical_parameters.number_of_elements}.pkl'
    else:
        filename = f'results/{model_name}_{criterion.type}_{criterion.threshold}_t={model.t_max}_dt={numerical_parameters.dt}_elements={numerical_parameters.number_of_elements}.pkl'


    simulation_result.save(filename)


def visualize_results(pickle_file_path, amount_graphs:int = 5):
    simulation_result = SimulationResult.load(pickle_file_path)
    #simulation_result.plot_criteria_norm()
    for i in range(0, len(simulation_result.q_in_time), int(len(simulation_result.q_in_time)/amount_graphs)-1):
        #simulation_result.plot_water_hight_at_index(i, save_data=True, savedata_path= '../images')
        simulation_result.plot_water_hight_at_index(i)

def gif_visualization(pickle_file_path):
    run_time = time.time()

    simulation_result = SimulationResult.load(pickle_file_path)

    path_name = pickle_file_path[:-4]

    if not os.path.exists(path_name):
        os.makedirs(path_name)

    args_list = [(i, True, path_name) for i in range(len(simulation_result.q_in_time))]

    with Pool() as pool:
        pool.starmap(simulation_result.plot_water_hight_at_index, args_list)

    filenames = [f'{path_name}/step_{i:04d}.png' for i in range(1, len(simulation_result.q_in_time))]

    with imageio.get_writer(f'{path_name}.gif', mode='I', duration=simulation_result.time_step) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    run_time = time.time() - run_time
    print(f'Time for one run: {run_time:.2f}s.')


if __name__ == "__main__":
    #simulation_run()
    #visualize_results('results/solitary_wave_U_X_PROTECTED_0.00015_t=50.0_dt=0.05_elements=400.pkl')  #('results/beji_battjes_W_X_PROTECTED_0.0005_t=50.0_dt=0.01_elements=400.pkl')
    gif_visualization('results/solitary_wave_GLOBAL_t=50.0_dt=0.05_elements=400.pkl')
