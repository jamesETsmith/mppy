"""
This examples creates an ethylene molecule in the planar geometry rotates the dihedral angle,
calculating the energy at each step, creating a 1D PES.

Author: James E. T. Smith <james.smith9113@gmail.com>
Date: 1/3/20
"""
import numpy as np
from pyscf import gto, scf
import matplotlib.pyplot as plt
import seaborn as sns

from mppy.mpMolecule import mpMolecule
from mppy.manipulation import rotate_dihedral

#
# Use mppy to create geometries
#
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
npts = 30
rotor = [2, 3]
phi = np.linspace(0, np.pi, num=npts)
rotated_xyz = [rotate_dihedral(xyz, 0, 1, phi_i, rotor) for phi_i in phi]

#
# For each geometry calculate/save the energy
#
energies = []
for i, geom in enumerate(rotated_xyz):
    mol = mpMolecule(geom, atom)
    pyscf_mol = gto.M(atom=mol.dump_pyscf_atom())
    mf = scf.RHF(pyscf_mol)
    energies.append(mf.scf())


#
# Plot Energy vs. Angle of Rotation
#
plt.figure()
sns.set_style("darkgrid")

plt.plot(phi * 180 / np.pi, energies, "o-")

plt.xlabel("Dihedral Angle in Degrees")
plt.ylabel("Energy (Ha)")
plt.tight_layout()
plt.savefig("figures/04_1D_PES.png", dpi=600)
