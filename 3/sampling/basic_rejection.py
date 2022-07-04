# Basic rejection sampling of a Student-t distribution. By basic is ment that
# the secondary sampled distribution is a uniform distribution, between 0 and
# the maximum of the distribution we're interested in.
# A Student-t distribution is chosen for the example because it's fairly
# difficult to sample from otherwise. The CDF requires the _2F_1 hypergeometric
# function, with no obvious inverse. (A Gaussian also has a CDF with no obvious
# inverse, but at least erf is common enough to work with.)
# A Student-t distribution is also chosen because it highlights many of the
# shortcomings of basic rejection sampling, and rejection sampling as a whole;
# more on this later.

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

n    = 10000 # number of samples
bins = 128   # number of histogram bins
N    = 1000  # function plot resolution

ups = 1 # DOF (Student-t parameter)

# A Student-t distribution has long 'wings' as they are called; it does not tend
# to a negligible probability very quickly. Because of this, a wide sampling
# range is necessary to include the wings, and the value of the PDF is much
# smaller than its maximum for most of the range. As a result, the sampling
# efficiency will be poor; more on this later.
t_range = [-20, 20]

# Knowing the maximum of a distribution is necessary for rejection sampling.
# This is a major issue of rejection sampling, because for real problems this is
# often very difficult or impossible to determine. Sometimes it's also much
# larger than the average of the PDF across the considered range. In such a case
# the sampling efficiency will be poor; more on this later.
f_max = special.gamma(.5*(ups+1))/np.sqrt(ups*np.pi)/special.gamma(.5*ups)
# Student-t distribution
def f(t):
    return f_max*(1+t**2/ups)**(-.5*(ups+1))

# Even with the wide range, everything farther and lower than the following
# cutoff probability is lost. Notice that this is not that small; if millions of
# samples are made, the samples are qualitatively highly improbable to have
# come from the Student-t distribution, i.e. they can be considered poor quality
# samples.
print(f'probability cutoff {100*f(t_range[1]):.3f}%')

# basic rejection sampling
s = np.empty(n) # samples
# A fast and high quality random number generator is of course key to any
# Monte-Carlo method. Not discussed here. Using a random seed here to show how
# the random nature of Monte-Carlo sampling; more on this later.
rng = np.random.default_rng()
i = 0 # sample count
j = 0 # attempt count
while i < n:
    t = rng.uniform(t_range[0], t_range[1])
    p = rng.uniform(0, f_max)
    if p < f(t):
        s[i] = t
        i += 1
    j += 1

# Sampling efficiency was mentioned earlier without definition. This is exactly
# what is ment: the probability a sample will get accepted. This should ideally
# be 1.
# (Technically the sample count over attempt count is only a quantity that
# approximates the true sampling efficiency; the sampling efficiency is not
# random, but whatever. The obvious experimental sampling efficiency is given as
# well.)
# The sampling efficiency for basic rejection sampling is trivial and need not
# be explained, it just follows from the geometric interpretation of rejection
# sampling.
# The fact that sampling efficiency isn't always 1 is by itself a major
# disadvantage of rejection sampling, let alone an efficiency near 0, as is the
# case here.
# Notice how the true sampling efficiency is different from the experimental
# one, and how the experimental sampling efficiency varies due to the new RNG
# seed every time the program is ran.
# Notice how the true sampling efficiency tends to be slightly higher than the
# experimental one. This is because it is assumed the integral of the PDF is 1
# in the considered range. This isn't the case due to the finite range. The fact
# that this is noticable again goes to show the range is not large enough.
eff = 1/(f_max*(t_range[1]-t_range[0]))
print(f'true         sampling efficiency {100*eff:.2f}%')
print(f'experimental sampling efficiency {100*i/j:.2f}%')

# plot
t = np.linspace(t_range[0], t_range[1], N)
plt.hist(s, bins=bins, density=True, color='k')
plt.plot(t, [f(t_) for t_ in t], 'r')
plt.plot(t_range, [f_max, f_max], '--r')
plt.xlim(t_range)
plt.ylabel('$p$')
plt.tight_layout()
plt.show()
