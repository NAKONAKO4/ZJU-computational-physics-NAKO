import os
import numpy as np
from Prepare import Prepare
from Utilities import Utilities


class InitializeSimulation(Prepare, Utilities):
    def __init__(self,
                box_dimensions,  # List - Angstroms
                cut_off, # Angstroms
                initial_positions=None,  # Array - Angstroms
                neighbor=1, # Integer
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.box_dimensions = box_dimensions
        self.cut_off = cut_off
        self.neighbor = neighbor
        self.step = 0 # initialize simulation step
        self.initial_positions = initial_positions
        self.nondimensionalize_units(["box_dimensions", "cut_off",
                                      "initial_positions"])

    def define_box(self):
        """Define the simulation box. Only 3D boxes are supported."""
        box_boundaries = np.zeros((3, 2))
        for dim, L in enumerate(self.box_dimensions):
            box_boundaries[dim] = -L / 2, L / 2
        self.box_boundaries = box_boundaries
        box_size = np.diff(self.box_boundaries).reshape(3)
        box_geometry = np.array([90, 90, 90])
        self.box_size = np.array(box_size.tolist() + box_geometry.tolist())