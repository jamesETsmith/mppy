"""
This examples creates an ethylene molecule in the planar geometry and then 
pyramidalized one of the CH2 groups.

Author: James E. T. Smith <james.smith9113@gmail.com>
Date: 1/2/20
"""

import matplotlib.pyplot as plt
import numpy as np

from mppy.mpMolecule import mpMolecule
from mppy.manipulation import pyramidalize

xyz = np.array(
    [
        [-1.71614101, -0.62152133, 0.00000000],
        [-1.18297727, -1.54922625, 0.00000000],
        [-2.78614101, -0.62152133, 0.00000000],
        [-1.04086671, 0.55345597, 0.00000000],
        [-1.57403045, 1.48116089, 0.00000000],
        [0.02913329, 0.55345597, 0.00000000],
    ]
)

atom = ["C", "H", "H", "C", "H", "H"]
mol = mpMolecule(xyz, atom)
mol.get_bonds_by_distance()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mol2 = pyramidalize(mol, 0, 1, 2, 3, np.pi / 2)
mol2.plot(ax)


ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)
plt.legend()
plt.show()
