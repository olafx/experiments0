'''
2D Airy pattern of two equal intensity objects at the Rayleigh resolving limit.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import j1, jn_zeros
from numpy import linspace, meshgrid, pi, sqrt, sin, cos, log

xl = 18
res = 1000
θ = pi/6

x = linspace(-xl, xl, res)
x, y = meshgrid(x, x)
d = jn_zeros(1, 1)[0]
x1 = sqrt((x-.5*d*cos(θ))**2+(y+.5*d*sin(θ))**2)
x2 = sqrt((x+.5*d*cos(θ))**2+(y-.5*d*sin(θ))**2)
I1 = (2*j1(x1)/x1)**2
I2 = (2*j1(x2)/x2)**2
img = plt.imshow(I1+I2, extent=(-xl, xl, -xl, xl), norm=colors.PowerNorm(1/log(xl)), cmap='Greys_r')
plt.xlim(-xl, xl)
plt.ylim(-xl, xl)
plt.colorbar(img, label='$I/I_0$')
plt.savefig('Airy_3.png', dpi=400)
