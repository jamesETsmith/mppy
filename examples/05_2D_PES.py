"""
This examples creates an ethylene molecule in the planar geometry rotates the dihedral angle
and pyramidalizes one C atom, calculating the energy at each step, creating a 2D PES.

Author: James E. T. Smith <james.smith9113@gmail.com>
Date: 1/3/20
"""
import numpy as np
from pyscf import gto, scf

from mppy.mpMolecule import mpMolecule
from mppy.manipulation import rotate_dihedral, pyramidalize

dim = 10


def z_func(xyz, atom, X, Y):
    rotor = [2, 3]

    energies = np.zeros_like(X)

    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            new_xyz = rotate_dihedral(xyz, 0, 1, X[x, y], rotor)
            new_xyz = pyramidalize(new_xyz, 0, 2, 3, 1, Y[x, y])
            mol = gto.M(atom=mpMolecule(new_xyz, atom).dump_pyscf_atom())
            mf = scf.RHF(mol).run()
            energies[x, y] = mf.e_tot

    return energies


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

phi = np.linspace(0, np.pi, num=dim)
tau = np.linspace(0, np.pi / 2, num=dim)
X, Y = np.meshgrid(phi, tau)
Z = z_func(xyz, atom, X, Y)

# savez saves a multiple arrays at once
# using kwargs allows them to be accessible by the kwargs
# when they are loaded
np.savez("data/05_data.npz", X=X, Y=Y, Z=Z)

