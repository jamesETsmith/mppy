"""
This examples creates an ethylene molecule in the planar geometry and then 
rotates about the C=C bond 90 degrees.

Author: James E. T. Smith <james.smith9113@gmail.com>
Date: 12/20/19
"""

import matplotlib.pyplot as plt
import numpy as np

from mppy.mpMolecule import mpMolecule


xyz = np.array(
    [
        [3.402, 0.773, -9.252],
        [4.697, 0.791, -8.909],
        [2.933, -0.150, -9.521],
        [2.837, 1.682, -9.258],
        [5.262, -0.118, -8.904],
        [5.167, 1.714, -8.641],
    ]
)
atom = ["C", "C", "H", "H", "H", "H"]
mol = mpMolecule(xyz, atom)
mol.get_bonds_by_distance()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mol.plot(ax)
plt.show()
