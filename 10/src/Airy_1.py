'''
1D Airy pattern with vertical lines showing the zeros.
'''

import matplotlib.pyplot as plt
from scipy.special import j1, jn_zeros
from numpy import linspace, arange, pi
from math import ceil

ZEROS = True
xl = 12
res = 1000

x = linspace(-xl, xl, res)
I = (2*j1(x)/x)**2
plt.plot(x, I, 'k')
if ZEROS:
    x0 = jn_zeros(1, max(1, xl//pi)) # upper limit
    for x0_ in x0:
        if x0_ < xl:
            plt.plot([-x0_, x0_], [0, 0], 'or')
plt.xlim(-xl, xl)
plt.xlabel(r'$k a \sin \theta$')
plt.ylabel('$I / I_0$')
plt.savefig('Airy_1.png', dpi=400)
