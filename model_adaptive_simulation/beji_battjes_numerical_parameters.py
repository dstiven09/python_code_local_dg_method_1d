class BejiBattjesNumericalParameters:

    def __init__(self) -> None:
        self.interpolation_order = 1
        self.max_iterations = 1000
        self.number_of_elements = 400
        self.t_max = 40.0
        self.dt = self.t_max / (self.number_of_elements - 1.)
        self.number_of_global_steps = 0
        self.discretization_order = 2
