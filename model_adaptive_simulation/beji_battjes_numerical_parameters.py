class BejiBattjesNumericalParameters:

    def __init__(self) -> None:
        self.interpolation_order = 1
        self.max_iterations = 10000
        self.number_of_elements = 400
        self.dt = 0.01 #(self.t_max) / (self.number_of_elements - 1.) #It needs revision
        self.number_of_global_steps = 10
        self.discretization_order = 2
