# Joukowsky airfoil via conformal map.

import numpy as np
import matplotlib.pyplot as plt

# these parameters are definitely not standard names
obtuseness = .02
width = .1

n = 100 # resolution

# points homogeneously distributed on circle
theta = np.linspace(0, 2*np.pi, n)
pos_1 = (1+width)*(np.cos(theta)+1j*np.sin(theta))
pos_1 += -width+obtuseness+width*1j

# apply conformal mapping to points
pos_2 = pos_1+1/pos_1

# plot
plt.plot(np.real(pos_2), np.imag(pos_2), 'k')
plt.axis('square')
plt.tight_layout()
plt.show()
