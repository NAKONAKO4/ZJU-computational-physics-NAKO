import numpy as np
from InitializeSimulation import InitializeSimulation


class Measurements(InitializeSimulation):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)