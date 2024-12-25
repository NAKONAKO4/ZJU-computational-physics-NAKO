from Measurements import Measurements
import numpy as np
import copy
import os


class MinimizeEnergy(Measurements):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)