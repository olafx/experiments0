# Smart rejection sampling of a Student-t distribution. By smart is ment that
# the proposal distribution is almost exactly identical to the target Student-t
# distribution, except much easier to sample from. This way the sampling
# efficiency is high.
# A Lorentzian distribution (or Cauchy distribution as known outside of physics) 
# is a good fit here for a proposal distribution, because it's similar to the
# target Student-t and has a simple quantile function, so sampling is easy; just
# run a uniform probability through the quantile function.

import sys
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

n    = 10000 # number of samples
bins = 128   # number of histogram bins
N    = 1000  # function plot resolution

ups   = 1   # DOF (Student-t parameter)
gamma = 1.2 # scale/HWHM (Lorentz parameter)
alpha = 1.2 # scales the proposal distribution so that it's no longer a PDF

# The sampling range can be orders of magnitudes larger here as in basic
# rejection sampling, because the efficiency will be much higher. If we have a
# mathematical guarantee that the target distribution will always be larger than
# the proposal distribution, it can even be made infinite.
# The plot uses the same range as before for clarity.
t_range      = [-1e3, 1e3]
t_range_plot = [-20, 20]

# Student-t distribution
f_max = special.gamma(.5*(ups+1))/np.sqrt(ups*np.pi)/special.gamma(.5*ups)
def f_t(t):
    return f_max*(1+t**2/ups)**(-.5*(ups+1))

# The distribution used to compare samples can't be normalized. If it was, it
# would either have to be an identical distribution, i.e. useless because we
# can't pull samples from it, or be smaller than the distribution we're
# interested in in some areas, which is not allowed. If alpha were 1.1 here, it
# would not work.
# scaled Lorentzian distribution (not a PDF, unless alpha = 1)
def f_L(x):
    return alpha/(np.pi*gamma)*gamma**2/(gamma**2+x**2)

# weak check to see if proposal > target everywhere
for t in np.linspace(t_range[0], t_range[1], n):
    if f_L(t) < f_t(t):
        raise RuntimeError('proposal distribution is smaller than target'
            'distribution somewhere')

# The reason to choose a Lorentzian is its simple quantile function, along with
# being very similar to the Student-t distribution. This will allow for high
# sampling efficiency.
# Lorentzian quantile function
def Q_L(p):
    return gamma*np.tan(np.pi*(p-.5))

# smart rejection sampling
s = np.empty(n) # samples
# Same random number generator, still random seed to show the randomness of
# Monte-Carlo sampling.
rng = np.random.default_rng()
i = 0
j = 0
while i < n:
    t = Q_L(rng.uniform(0, 1))
    if t < t_range[0] or t > t_range[1]:
        continue
    p = rng.uniform(0, f_L(t))
    if p < f_t(t):
        s[i] = t
        i += 1
    j += 1

# The true sampling efficiency if we considered an infinite range would be the
# ratio of the integrals, which is trivially 1/alpha, since the target and
# proposal distributions are PDFs, other than the alpha factor in front of the
# proposal. We don't consider an infinite range, so it's not exactly correct.
# That's why the experimental sampling efficiency might be slightly differnet on
# average.
# Notice how we went from <8% efficiency to >83% efficiency, and the parameters
# alpha and gamma aren't even that optimized. The important conclusion is that
# in this case, smart rejection sampling is fine, and more advanced Markov chain
# Monte-Carlo methods are not necessary.
print(f'true         sampling efficiency {100/alpha:.2f}%')
print(f'experimental sampling efficiency {100*i/j:.2f}%')

# for the plot it's necessary to remove samples outside of the plot range,
# because otherwise the bin width becomes too large
s = s[~(s < t_range_plot[0])]
s = s[~(s > t_range_plot[1])]

# plot
t = np.linspace(t_range_plot[0], t_range_plot[1], 1000)
plt.plot(t, [f_t(t_) for t_ in t], 'r', label='target Student-$t$')
plt.plot(t_range, [f_max, f_max], '--r')
plt.plot(t, [f_L(t_) for t_ in t], 'b', label='proposal Lorentzian')
plt.hist(s, bins=bins, density=True, color='k')
plt.xlim(t_range_plot)
plt.ylabel('$p$')
plt.legend()
plt.tight_layout()
plt.show()
