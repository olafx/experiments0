# Metropolis-Hastings algorithm for Markov chain Monte-Carlo.
# e^(-x^4) is sampled. Its integral is 2Gamma(5/4) and its maximum is 1, but
# neither of these are needed for sampling. A normal distribution is used for
# suggesting new samples. The variance is adjusted through to optimize sampling
# efficiency during the burn-in period. More on this later.

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

n    = 10000 # number of samples
bins = 64    # number of histogram bins
N    = 1000  # function plot resolution

n_b  = 10000 # burn-in length
n_a  = 64    # length for approximate momentary sampling efficiency
r    = 3     # plot range is -r to r

var_regulate_rate = .001 # aggressiveness of variance adjustment
var_init          = 2    # initial normal distribution variance

# The unnormalized target distribution.
def f(x):
    return np.exp(-x**4)
c = 2*special.gamma(1.25)
# The normalized target distribution. (Only needed for plotting.)
def p(x):
    return f(x)/c

# Markov chain Monte-Carlo sampling starts here.

# Same random number generator, still with a runtime seed to show the randomness
# of Monte-Carlo sampling.
rng = np.random.default_rng() 

# Here begins the burn-in phase. The variance is adjusted throughout, more on
# this later.

# Using the center for the initial state.
x = 0          # initial state
i = 0          # sample count
j = 0          # attempt count
var = var_init # normal distribution variance

# The SMA (simple moving average) of acceptance rate is used for regulating
# variance during burn-in. This requires a recent history.
hist = np.empty(n_a)
# Keeping a history of the SMA and variance for plotting.
sma_hist = []
var_hist = [var]

while i < n_b:
    accept = False
    y = rng.normal(x, var) # candidate
    r_f = f(y)/f(x)
    # Since g is symmetric, i.e. since this is Metropolis sampling,
    # r_g = g(x|y)/g(y|x) = 1. For completeness it's not removed from the
    # algorithm.
    r_g = 1
    if rng.uniform() < min(1, r_f*r_g):
        i += 1
        accept = True
        x = y
    if j < n_a: # initial history filling
        hist[j] = 1 if accept else 0
    else:
        sma = np.sum(hist)/n_a
        sma_hist.append(sma)
        # Mathematically, maximum Metropolis-Hastings performance is achieved
        # for 1D functions when the acceptance rate is 50%. We use the SMA of
        # the acceptance rate as a measurement of the momentary acceptance rate,
        # and regulate the variance with the goal of keeping the acceptance rate
        # around 50%. This can be done by multiplying the variance by some sort
        # of factor that is >1 when the acceptance rate is too high, and <1 if
        # it's too low. Moreover, it makes sense that the severity of this
        # change reflects the severity of the acceptance rate error, so the
        # factor should also be proportional to the acceptance rate error.
        r_acc = 2*sma # ratio of SMA acceptance rate to ideal acceptance rate
        # Multiplying variance by r would be a natural first guess, but this
        # change is too sudden and the variance becomes unstable, so it needs to
        # be weakened a bit, i.e. pulled towards one. This can be done by
        # separating r into 1+(r-1) and multiplying r-1 by some factor <1.
        # This factor needs to be large enough so that the variance can be
        # adjusted sufficiently during burn-in, but not too large or it will
        # become unstable. Unfortunately I don't know of a rigorous way to
        # calculate a good factor, so I leave it a parameter. It must be related
        # to n_a however since the SMA reacts more slowly for larger n_a,
        # requiring more patience, and thus a smaller factor.
        var *= 1+var_regulate_rate*(r_acc-1)
        var_hist.append(var)
        # Move up the history.
        hist[:] = np.append(hist[1:], 1 if accept else 0)
    j += 1

print(f'variance before burn-in {var_init:.2f}')
print(f'variance after burn-in {var:.2f}')
# Unlike the variance, the acceptance rate varies widely due to the random
# nature of Monte-Carlo sampling. So a mean must be calculated before and after.
print('average acceptance rate during burn-in (initial 10%) '
    f'{np.mean(sma_hist[:n_b//10]):.2f}')
print('average acceptance rate during burn-in (final 10%) '
    f'{np.mean(sma_hist[-n_b//10:]):.2f}')
# The latter should be around 50%.

# Here begins the actual sampling phase, using whatever variance we ended up
# with during burn-in, and whatever sample x we ended with.
s = np.empty(n) # samples
s[0] = x
i = 1 # sample count
j = 1 # attempt count
while i < n:
    y = rng.normal(x, var) # candidate
    r_f = f(y)/f(x)
    r_g = 1
    if rng.uniform() < min(1, r_f*r_g):
        s[i] = y
        i += 1
    j += 1

# Plotting.

x = np.linspace(-r, r, N)

plt.figure('distributions')
plt.plot(x, [p(x_) for x_ in x], 'r', label='target $e^{-x^4}$')
plt.hist(s, bins=bins, density=True, color='k')
plt.xlim(-r, r)
plt.xlabel('$x$')
plt.ylabel('$f$')
plt.legend()
plt.tight_layout()

plt.figure('acceptance rate during burn-in')
plt.plot(sma_hist, 'k')
plt.xlabel('attempts during burn-in')
plt.ylabel(f'acceptance rate (SMA of {n_a})')
plt.xlim(0, len(sma_hist))
plt.ylim(0, 1)
plt.tight_layout()

plt.figure('variance during burn-in')
plt.plot(var_hist, 'k')
plt.xlabel('attempts during burn-in')
plt.ylabel('proposal normal variance $\sigma^2$')
plt.xlim(0, len(var_hist))
plt.tight_layout()

plt.show()
