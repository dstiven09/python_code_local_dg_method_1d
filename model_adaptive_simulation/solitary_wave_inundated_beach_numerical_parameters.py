class SolitaryWaveInundatedBeachNumericalParameters:

    def __init__(self) -> None:
        self.interpolation_order = 1
        self.max_iterations = 1000
        self.number_of_elements = 400
        self.dt = 6. / (self.number_of_elements - 1.)
        self.number_of_global_steps = 10
        self.discretization_order = 2
