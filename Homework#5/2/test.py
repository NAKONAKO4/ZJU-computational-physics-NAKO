import numpy as np
from itertools import product

states = np.array(list(product([-1, 1], repeat=25)))
print(states)