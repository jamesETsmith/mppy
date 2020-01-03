import numpy as np
import pytest

from mppy.mpMolecule import mpMolecule
from mppy import manipulation as manip

npt = np.testing


def test_er():
    v1 = np.array([0, 0, 1.0])
    v2 = np.array([0, 1.0, 0])

    rot = manip.align_vectors(v1, v2)
    v3 = np.dot(rot.T, v2)

    npt.assert_almost_equal(v3, v1)


def test_dihedral_rotation():
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
    rotor = np.array([0, 1, 2])
    npts = 2
    rotated_mols = manip.rotate_dihedral(0, 3, np.pi, npts, rotor, xyz)

    npt.assert_equal(npts + 1, rotated_mols.shape[0])

    # Make sure that molecules that aren't in the rotor are stationary
    npt.assert_equal(rotated_mols[0, 0], rotated_mols[1, 0])
    npt.assert_equal(rotated_mols[0, 0], rotated_mols[2, 0])

    # Make sure bonds are still the same length
    b1 = np.linalg.norm(rotated_mols[0, 1] - rotated_mols[0, 0])
    b2 = np.linalg.norm(rotated_mols[1, 1] - rotated_mols[1, 0])
    b3 = np.linalg.norm(rotated_mols[2, 1] - rotated_mols[2, 0])
    npt.assert_equal(b1, b2)
    npt.assert_equal(b3, b2)

    # Make sure the Hs rotated and rotated back properly
    # npt.assert_equal(rotated_mols[0, 1], rotated_mols[1, 2])
    # npt.assert_equal(rotated_mols[1, 1], rotated_mols[0, 2])

    npt.assert_almost_equal(rotated_mols[0, 1], rotated_mols[2, 1])
    npt.assert_almost_equal(rotated_mols[0, 2], rotated_mols[2, 2])


def test_pyramidalize():
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
    xyz -= xyz[0]

    atom = ["C", "H", "H", "C", "H", "H"]
    mol = mpMolecule(xyz, atom)

    mol2 = manip.pyramidalize(mol, 0, 1, 2, 3, 1 / 3 * np.pi)
    mol3 = manip.pyramidalize(mol2, 0, 1, 2, 3, -1 / 3 * np.pi)

    npt.assert_almost_equal(mol.get_xyz(), mol3.get_xyz())


def test_pyscf_parser():
    atom = """O 0. 0. 0.
        H 0. 1. 0.
        H 0. 0. 1.
        """

    mol = manip.parse_pyscf_atom(atom)
    npt.assert_equal(mpMolecule, type(mol))
