import matplotlib.pyplot as plt
import numpy as np

from mppy import mpMolecule

atoms = ["C", "H", "H", "H", "C", "H", "C", "H", "C", "H", "H", "H"]
xyz = np.array(
    [
        [-4.78885668, 1.36034493, 0.00000000],
        [-4.43220225, 0.35153493, 0.00000000],
        [-4.43218384, 1.86474312, -0.87365150],
        [-5.85885668, 1.36035811, 0.00000000],
        [-4.27551446, 2.08630120, 1.25740497],
        [-3.66047964, 1.56454028, 1.96053917],
        [-4.60006716, 3.38370673, 1.47642888],
        [-4.24040385, 3.88279822, 2.35190018],
        [-5.48474662, 4.14217805, 0.46962030],
        [-5.63159701, 5.14610950, 0.80940938],
        [-5.00532224, 4.15589711, -0.48686496],
        [-6.43200027, 3.65151711, 0.38678096],
    ]
)

# mol = mpMolecule(xyz, atoms)

npts = 24
rotor = np.arange(6)
rotated_xyz = mpMolecule.rotate_dihedral(4, 6, 2 * np.pi / npts, npts, rotor, xyz)

# ffmpeg -framerate 24 -i %d.png test.mpeg
n_repeats = 10
i = 0
for rxyz in rotated_xyz:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20.0, azim=45)
    mol = mpMolecule.mpMolecule(rxyz, atoms)
    mol.plot(ax)

    # Slowing down the framerate in ffmpeg can be a pain
    for j in range(n_repeats):
        plt.savefig("figures/%d" % i)
        i += 1

    plt.close()
