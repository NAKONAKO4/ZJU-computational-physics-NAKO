import numpy as np
import copy
from Measurements import Measurements

import warnings
warnings.filterwarnings('ignore')


class MonteCarlo(Measurements):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)