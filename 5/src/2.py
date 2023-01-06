'''
Bifurcation diagram of the logistic map f(x) = rx(1-x), together with its
Lyapunov exponents.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np

# number of stabilization iterations
n_i = 1000
# number of plot iterations
n_p = 30
# r range, number of r values
range_r = 3.7, 3.8
n_r = int(1e5)
# initial value
x0 = .2

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(2, figsize=(14, 8))

r = np.linspace(*range_r, n_r)
x = np.empty((n_p, n_r))
x[0] = x0
l = np.log(np.abs(r*(1-2*x[0]))) # f'(x) = r(1-2x)
for i in range(n_i+1):
    x[0] = r*x[0]*(1-x[0])
    l += np.log(np.abs(r*(1-2*x[0])))
for p in range(n_p-1):
    x[p+1] = r*x[p]*(1-x[p])
    l += np.log(np.abs(r*(1-2*x[p+1])))
l = l/(1+n_i+n_p)
for p in range(n_p):
    ax[0].plot(r, x[p], ',', c='black', ms=.01)

ax[0].set_xlim(*range_r)
ax[0].set_ylim((0, 1))
ax[0].set_xlabel('$r$', fontsize=14)
ax[0].set_ylabel(f'$x_{{{n_i+1}}}$ to $x_{{{n_i+n_p}}}$', fontsize=14)

ax[1].set_xlim(*range_r)
ax[1].set_ylim((-1, 1))
ax[1].set_xlabel('$r$', fontsize=14)
ax[1].set_ylabel(f'$\lambda({x0})$')
ax[1].grid()
ax[1].plot(range_r, (0, 0), c='red', lw=.5)
ax[1].plot(r, l, c='black', lw=.5)

plt.tight_layout()
plt.savefig('bifurcation.png', dpi=300, bbox_inches='tight')
