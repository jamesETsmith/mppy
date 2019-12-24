import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

la = np.linalg
from mendeleev import element


class mpMolecule:
    """
    Wrapper for molecules.
    """

    def __init__(self, xyz, atom, mass=np.array([]), bonds=np.array([])):
        """Init. mpMolecule wrapper.
        
        Parameters
        ----------
        xyz : `np.ndarray`
            2D array of coordinates (default in angstroms
        atom : [type]
            1D array of atomic numbers.
        mass : [type], optional
            1D array of masses, by default np.array([])
        bonds : [type], optional
            2D array of bonds, by default np.array([])
        """
        self._xyz = xyz
        self._atom = np.zeros(xyz.shape[0], dtype=np.int)

        # If symbols given, convert to atomic number
        if isinstance(atom[0], int) != True:
            for i, ai in enumerate(atom):
                # self._atom[i] = getattr(elements, ai).number
                self._atom[i] = element(ai).atomic_number

        # Set masses as isotopically weighted avg if none given
        if mass.size == 0:
            self._mass = np.zeros(self._atom.shape)
            for i, el in enumerate(self._atom):
                # self._mass[i] = elements[el].mass
                self._mass[i] = element(int(el)).atomic_weight
        else:
            self._mass = mass

        # Bonds
        self._bonds = bonds

    def plot(self, ax, show_legend=False):
        """Plot the molecule on the given axis"""

        # Set the color array for the atoms
        colors = []
        size = []
        symbs = []

        for an in self._atom:
            colors.append(element(int(an)).cpk_color)
            size.append(element(int(an)).atomic_radius * 10)
            symbs.append(element(int(an)).symbol)

        for i in range(self._bonds.shape[0]):
            xs = np.zeros((2,))
            ys = np.zeros((2,))
            zs = np.zeros((2,))

            xs[0] = self._xyz[self._bonds[i, 0]][0]
            xs[1] = self._xyz[self._bonds[i, 1]][0]

            ys[0] = self._xyz[self._bonds[i, 0]][1]
            ys[1] = self._xyz[self._bonds[i, 1]][1]

            zs[0] = self._xyz[self._bonds[i, 0]][2]
            zs[1] = self._xyz[self._bonds[i, 1]][2]

            ax.plot(xs, ys, zs, c="darkgray", linewidth=5, zorder=1)

        # ax.scatter(self._xyz[:,0], self._xyz[:,1], self._xyz[:,2],
        #     c=colors,s=size, depthshade=False, zorder=10, label=symbs)

        # Plot the atoms one at a time to label them properly
        for i in range(self._xyz.shape[0]):
            # If the element has already been labelled, don't label it again
            if symbs[i] in symbs[:i]:
                ax.scatter(
                    self._xyz[i, 0],
                    self._xyz[i, 1],
                    self._xyz[i, 2],
                    c=colors[i],
                    s=size[i],
                    depthshade=False,
                    zorder=10,
                )
            else:
                ax.scatter(
                    self._xyz[i, 0],
                    self._xyz[i, 1],
                    self._xyz[i, 2],
                    c=colors[i],
                    s=size[i],
                    depthshade=False,
                    zorder=10,
                    label=symbs[i],
                )

        if show_legend:
            legend = ax.legend(labelspacing=2)  # Prevent overlapping symbols
            legend.get_frame().set_facecolor("#00FFCC")
            legend.get_frame().set_alpha(1)

    def get_bonds_by_distance(self, cutoff=1.6):
        """Determine the bonded atoms by a distance provied. Cutoff is inclusive.
        
        Parameters
        ----------
        cutoff : float, optional
            Bond length cutoff in Angstroms, by default 1.6
        """
        bonds = []
        for i in range(self._xyz.shape[0]):
            for j in range(i + 1, self._xyz.shape[0]):
                r = la.norm(self._xyz[i] - self._xyz[j])
                if r <= cutoff:
                    if self._atom[i] == self._atom[j] and self._atom[i] == 1:
                        continue
                    else:
                        bonds.append([i, j])

        self._bonds = np.array(bonds)
