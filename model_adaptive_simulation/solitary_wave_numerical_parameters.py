class SolitaryWaveNumericalParameters:

    def __init__(self) -> None:
        self.interpolation_order = 1
        self.max_iterations = 10000
        self.number_of_elements = 400
        self.dt = 0.05 #self.t_max / (self.number_of_elements - 1.)
        self.number_of_global_steps = 0
        self.discretization_order = 2
