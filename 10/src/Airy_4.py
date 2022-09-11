'''
2D Airy pattern of multiple variable intensity objects.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.special as special
import numpy as np

xl = 30
xl_dev = 6
n = 40

res = 1000
scale_power = .4

pos = np.random.normal(0, xl_dev, (n, 2))
I0 = np.random.exponential(1, n)
Imax = np.max(I0)

x = np.linspace(-xl, xl, res)
x, y = np.meshgrid(x, x)
d = special.jn_zeros(1, 1)[0]
I = np.zeros([res, res])
for i in range(n):
    x_ = np.sqrt((x-pos[i][0])**2+(y-pos[i][1])**2)
    I += I0[i]/Imax*(2*special.j1(x_)/x_)**2
img = plt.imshow(I, extent=(-xl, xl, -xl, xl), norm=colors.PowerNorm(scale_power), cmap='Greys_r')
plt.xlim(-xl, xl)
plt.ylim(-xl, xl)
plt.colorbar(img, label='$I/I_{max}$')
plt.savefig('Airy_4.png', dpi=400)
