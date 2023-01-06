'''
2D Airy pattern with highlighted zeros.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import j1, jn_zeros
from numpy import linspace, meshgrid, pi, sqrt, sin, cos, log

ZEROS = True
xl = 12
res = 1000

x = linspace(-xl, xl, res)
x, y = meshgrid(x, x)
x = sqrt(x**2+y**2)
I = (2*j1(x)/x)**2
img = plt.imshow(I, extent=(-xl, xl, -xl, xl), norm=colors.PowerNorm(1/log(xl)), cmap='Greys_r')
if ZEROS:
    x0 = jn_zeros(1, max(1, xl*sqrt(2)//pi)) # upper limit
    θ = linspace(0, 2*pi, res)
    for x0_ in x0:
        if x0_ < xl*sqrt(2): # include corners
            plt.plot(x0_*sin(θ), x0_*cos(θ), 'r', lw=.5)
plt.xlim(-xl, xl)
plt.ylim(-xl, xl)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'$\sqrt{x^2+y^2} = ka\sin\theta$')
plt.colorbar(img, label='$I/I_0$')
plt.savefig('Airy_2.png', dpi=400)
