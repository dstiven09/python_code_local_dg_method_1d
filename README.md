# Model-adaptive shallow water equations

This package contains classes to solve the shallow water equations with a model-adaptive solver. The solver uses a local criterion to determine wether to compute the solution using the computationally expensive non hydrostatic or the computationally inexpensive hydrostatic model. Further details about the solver can be found in [Leila Wegener's master's thesis](literature/masters_thesis_leila_wegener.pdf).
This package also contains a refactor of some test cases written by both Leila Wegener and Anja Jeschke. The main purpose of the refactor was to make the code more readable and reuseable in order to test the model-adaptive solver in multiple test cases.

## Structure

The code by Anja Jeschke is contained in `nonhydrostatic`. It consists of the old version of the test cases and classes to set up solving the shallow water equations.
The folder `model_adaptive_simulation` contains all files for the model-adaptive test cases. Each test case requires two files to be configured. The first one is the `*_model.py` file and the second one is the `*_numerical_parameters.py` file. In the `*_model.py` file all model parameters, initial and boundary conditions are configured.
The `*_numerical_parameters.py` contains all parameters for the solver such as step size or number of elements for the discontinuous Galerkin solver.
The class `ModelAdaptiveTestCase` in `model_adaptive_test_case.py` implements the solver for a abstract `Model` and `NumericalParameters`.
The class `ProjectionCriterion` in `projection_criterion.py` contains an enum of all possible projection criteria. It also contains methods to calculate the criteria and plot them for visualization.
The class `SimulationResult` in `simulation_result.py` saves all data required for a plotting after a simulation run. This class also provides methods for visualization and saving results.

## Usage

To run the solver for a given test case one of the predefined test cases can be used or a new test case can be implemented.

### Setting up a new test case

To create a new test case a new `*_model.py` and have to be created.
In particular, in `*_model.py` the methods `boundary_values` and `initial_condition` have to be implemented. Further, the following parameters have to be set in the `__init__` method:

- u_0
- A
- h_0
- still_water_depth
- t_max
- x_min
- x_max
- number_of_unknowns
- gravitational_constant
- is_non_linear_non_hydrostatic
- swnl
- B
- has_periodic_boundary

In `*_numerical_parameters.py` the following parameters have to be set:

- interpolation_order
- max_iterations
- number_of_elements
- dt
- number_of_global_steps
- discretization_order

### Running a test case

When both `*_model.py` and `*_numerical_parameters.py` are configured correctly, they can be passed as arguments to the `ModelAdaptiveTestCase` class. Here is an example for the solitary wave test case:

```Python
from solitary_wave_model import SolitaryWaveModel
from solitary_wave_numerical_parameters import SolitaryWaveNumericalParameters
from model_adaptive_test_case import ModelAdaptiveTestCase

model = SolitaryWave()
numerical_parameters = SolitaryWaveNumericalParameters()
test_case = ModelAdaptiveTestCase(model, numerical_parameters)
```

To run the a test case you can simply specify a projection criterion and a threshold and use the `run` method:

```Python
from projection_criterion import ProjectionCriterion, ProjectionCriterionType

criterion = ProjectionCriterion(ProjectionCriterionType.GLOBAL, 0.000125)
simulation_result = test_case.run(criterion)
```

The run method returns a `SimulationResult` which can be saved to a specified path as a pickle file:

```Python
simulation_result.save('results/simple_beach.pkl')
```

A result file can be loaded as follows:

```Python
from simulation_result import SimulationResult

simulation_result = SimulationResult.load('results/simple_beach.pkl')
```

### Visualizing results

To visualize a test case two methods have been implemented for `SimulationResult`:

- `plot_criteria_norm`
- `plot_criteria_at_time_index`

The method `plot_criteria_norm` plots the norm of each criterion at each time step. The method `plot_criteria_at_time_index` plots each criterion over the range of `x` for a given time step index. Note that an out of bounds error may occur if an index greater than the total number of time steps is chosen.

# Known issues

Currently, the `BejiBattjes` test case produces numerical instabilities after the wave hits the submerged obstacle. At this point the cause for these instabilities in unknown.
![Instabilities](images/criteria_norm_1.5.png)

# Set-up

This set-up has been tested for Python 3.8 and Python 3.9.

First clone the repository and move into the directory:

Then create a new virtual python environment to handle the dependencies.

```
python3 -m venv venv
```

Activate the virtual environment

```
source venv/bin/activate
```

and install all the dependencies

```
pip install -r requirements.txt
pip install -e nonhydrostatic
```

Note that you can deactivate the virtual environment by using

```
deactivate
```

To test if the setup has been done correctly you can run

```
python model_adaptive_simulation/main.py
```
