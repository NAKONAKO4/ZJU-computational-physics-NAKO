import numpy as np
from Prepare import Prepare
from pint import UnitRegistry
ureg = UnitRegistry()

# Define atom number of each group
nmb_1, nmb_2= [2, 3]
# Define LJ parameters (sigma)
sig_1, sig_2 = [3, 4]*ureg.angstrom
# Define LJ parameters (epsilon)
eps_1, eps_2 = [0.2, 0.4]*ureg.kcal/ureg.mol
# Define atom mass
mss_1, mss_2 = [10, 20]*ureg.gram/ureg.mol

# Initialize the prepare object
prep = Prepare(
    ureg = ureg,
    number_atoms=[nmb_1, nmb_2],
    epsilon=[eps_1, eps_2], # kcal/mol
    sigma=[sig_1, sig_2], # A
    atom_mass=[mss_1, mss_2], # g/mol
)

# Test function using pytest
def test_atoms_epsilon():
    expected = np.array([1., 1., 2., 2., 2.])
    result = prep.atoms_epsilon
    assert np.array_equal(result, expected), \
    f"Test failed: {result} != {expected}"
    print("Test passed")

# In the script is launched with Python, call Pytest
if __name__ == "__main__":
    import pytest
    pytest.main(["-s", __file__])