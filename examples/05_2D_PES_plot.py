"""
This examples reads the data from `05_2D_PES.py` and plots them as a surface plot.

Author: James E. T. Smith <james.smith9113@gmail.com>
Date: 1/3/20
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#
# Read in data
#
data = np.load("data/05_data.npz")
X = data["X"]
Y = data["Y"]
Z = data["Z"]

#
# Plot Energy vs. Angle of Rotation
#
plt.figure()
sns.set_style("darkgrid")
ax = plt.gca(projection="3d")

ax.plot_surface(X, Y, Z, cmap="winter")

ax.set_xlabel("$\phi$ (Rad.)")
ax.set_ylabel(r"$\tau$ (Rad.)")
ax.set_zlabel("Energy (Ha)")
plt.tight_layout()
plt.savefig("figures/05_2D_PES.png", dpi=600)
plt.show()
