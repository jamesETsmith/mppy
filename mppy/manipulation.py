import numpy as np

from mppy.mpMolecule import mpMolecule

la = np.linalg


def align_vectors(v1, v2):
    """Euler-Rodrigues rotation of vector 1 to align with vector 2.
    
    Parameters
    ----------
    v1 : `np.ndarray`
        Vector that will be rotated
    v2 : `np.ndarray`
        Vector that we will rotate to (i.e. we will make v1 || to v2)
    
    Returns
    -------
    `np.ndarray`
        3x3 rotation matrix that should be applied to v1.
    
    Raises
    ------
    AssertionError
        [description]
    """

    # Vector we will rotate about
    k = np.cross(v1, v2)
    k /= la.norm(k)

    # Angle we need to rotate
    th = np.arccos(np.dot(v1, v2) / (la.norm(v1) * la.norm(v2)))

    # # Euler/Rodrigues params
    # # See https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    # a = np.cos(th / 2.0)
    # b = k[0] * np.sin(th / 2.0)
    # c = k[1] * np.sin(th / 2.0)
    # d = k[2] * np.sin(th / 2.0)

    # print("CHECK %f = 1" % (a ** 2 + b ** 2 + c ** 2 + d ** 2))

    # if 1 != (a ** 2 + b ** 2 + c ** 2 + d ** 2):
    #     raise AssertionError("Something went wrong with rotation.")

    # r = np.array(
    #     [
    #         [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
    #         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
    #         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
    #     ]
    # )

    return rotate(k, th)


def rotate(k: np.ndarray, th: float) -> np.ndarray:
    """Rotation around an axis k using Euler-Rodrigues parameterization for rotation.
    
    Parameters
    ----------
    k : np.ndarray
        axis to rotate around
    th : float
        angle of rotation in radians
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """

    # Make sure k is a unit vector
    k /= la.norm(k)

    # Euler/Rodrigues params
    # See https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    a = np.cos(th / 2.0)
    b = k[0] * np.sin(th / 2.0)
    c = k[1] * np.sin(th / 2.0)
    d = k[2] * np.sin(th / 2.0)
    r = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )

    return r


def rotate_dihedral(
    p1: int, p2: int, th: float, npts: int, rotor: np.ndarray, xyz: np.ndarray
):
    """Rotate dihedral angle in molecule. Dihedral bond is always aligned to
    z-axis. Returns the original geometry + npts new rotated geometries.
    
    Parameters
    ----------
    p1 : int
        Index of pivot atom 1 in xyz array.
    p2 : int
        Index of pivot atom 2 in xyz array.
    th : float
        Angle to rotate rotor by.
    npts : int
        Number of rotations to perform.
    rotor : `np.ndarray`
        Array of indices of atoms that will be rotated.
    xyz : `np.ndarray
        2D ndarray of atom coordinates.
    
    Returns
    -------
    `np.ndarray`
        3D ndarray of rotated atom coordinates.
    """

    # Translate pivot 1 to the origin.
    xyz -= xyz[p1]

    # Rotate to align dihedral bond along z-axis.
    v1 = xyz[p2]
    v2 = np.array([0, 0, 1.0])
    r_mat = align_vectors(v1, v2)
    xyz = np.einsum("ij,kj->ik", xyz, r_mat)

    #
    rotated_xyz = np.zeros((npts + 1, xyz.shape[0], xyz.shape[1]))
    rotated_xyz[0, :, :] = xyz

    for i in range(1, npts + 1):
        rotated_xyz[i, :, :] = rotated_xyz[i - 1, :, :]
        rtr = rotated_xyz[i, rotor, :]
        r = rotate(v2, th)
        rtr = np.einsum("ij,kj->ik", rtr, r)
        rotated_xyz[i, rotor, :] = rtr

    return rotated_xyz


def pyramidalize(mol: mpMolecule, a1: int, a2: int, a3: int, a4: int, tau: float):
    """Approximately pyramidalize a sp2 atom (a1) in mol. Tau must by < np.pi.
    
    Parameters
    ----------
    mol : mpMolecule
        The molecule.
    a1 : int
        The central atom around which the pyramidalization takes place.
    a2 : int
        One of the ligands that will be rotated.
    a3 : int
        The second ligand that will be rotated.
    a4 : int
        The third atom bonded to the central atom (used to calcualate rotation).
    tau : float
        The angle to rotate.
    
    Returns
    -------
    mpMolecule
        The pyramidalized molecule.
    
    Raises
    ------
    ValueError
        Raised if |tau| > np.pi.
    """

    if abs(tau) > np.pi:
        raise ValueError("|tau| should be less than np.pi/2")

    # a1 is the pivot and remains in the same loc along with rest of molecule
    # a2 and a3 are rotated relative to the a1-a2-a3 plane
    # Move the pivot atom to the origin
    shifted_xyz = mol.get_xyz()
    shifted_xyz -= shifted_xyz[a1]
    v_plane = np.cross(shifted_xyz[a2], shifted_xyz[a3])
    v14 = shifted_xyz[a4]
    v_cross = np.cross(v_plane, v14)
    rot = rotate(v_cross, tau)

    rotated_xyz = shifted_xyz.copy()

    # TODO this should be applied to the connected substructures
    # not just the connecting atoms
    rotated_xyz[a2] = shifted_xyz[a2].dot(rot)
    rotated_xyz[a3] = shifted_xyz[a3].dot(rot)

    return mpMolecule(rotated_xyz, mol.get_atoms())


def parse_pyscf_atom(atom):
    '''
    Parses the mol.atom attribute of PySCF molecule object to get the
    coordinates and atoms and returns a Molecule object.
    .. note::
        Currently can only parse when mol.atom has the following form:
        mol.atom = """
        O 0. 0. 0.
        H 0. 1. 0.
        H 0. 0. 1.
        """
    atom (string): atom attribute of PySCF mol object
    '''

    atom_split = atom.split()
    atoms = []
    xyz = np.zeros((int(len(atom_split) / 4), 3))

    for i in range(xyz.shape[0]):
        atoms.append(atom_split[4 * i])
        xyz[i, 0] = float(atom_split[4 * i + 1])
        xyz[i, 1] = float(atom_split[4 * i + 2])
        xyz[i, 2] = float(atom_split[4 * i + 3])

    return mpMolecule(xyz, atoms)

