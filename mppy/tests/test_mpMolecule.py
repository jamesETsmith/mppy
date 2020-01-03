import pytest
import numpy as np
import matplotlib.pyplot as plt

from mppy.mpMolecule import mpMolecule

npt = np.testing


def setup():
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
    return mol


def test_mpMolecule_def():
    mol = setup()

    npt.assert_equal(mol._atom, [6, 6, 1, 1, 1, 1])
    npt.assert_equal(mol._mass, [12.011] * 2 + [1.008] * 4)


def test_mpMolecule_bonds():
    mol = setup()
    mol.get_bonds_by_distance()

    bonds = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [1, 5]])
    npt.assert_equal(bonds, mol._bonds)


def test_mpMolecule_plot():
    mol = setup()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mol.plot(ax)
    # plt.show()
    plt.close()


def test_mpMolecule_dump_pyscf():
    mol = setup()
    pyscf_atom = mol.dump_pyscf_atom()

    ans = [
        [6, [3.402, 0.773, -9.252]],
        [6, [4.697, 0.791, -8.909]],
        [1, [2.933, -0.150, -9.521]],
        [1, [2.837, 1.682, -9.258]],
        [1, [5.262, -0.118, -8.904]],
        [1, [5.167, 1.714, -8.641]],
    ]

    npt.assert_equal(pyscf_atom, ans)
