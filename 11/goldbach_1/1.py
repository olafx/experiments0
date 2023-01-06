'''
An algorithm to calculate the number of ways i can be represented as the sum
of 2 primes for a range of even integers i. Goldbach's conjecture says there
should be at least one way for every i > 3.

The algorithm works by first calculating the relevant primes, then subtracting
each relevant prime p[j] from i. It effectively remembers what the last prime
i-p[j] was. The next prime will be smaller than the previous; often it is just
i-p[j-1]. It is smart to look for this specific prime candidate, and rejects it
right when it has become clear that this prime is skipped.

Some other optimizations are made also, like no even prime solutions existing
for anything other than i = 4.
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

n = 100 # The range is [2, n].
DEBUG = True
PRINT_MULTIPLICITIES = True

def debug(line: str):
    if DEBUG:
        print(line)

m = np.zeros((n//2), dtype=np.uint)

p = np.array(list(sp.sieve.primerange(2, n-2+1)))

# p[a] is the largest prime that is at most i/2.
# p[b] is the largest prime that is at most i-2.
a = 0 
b = 0 

# i = 2 is the special case where there are no such primes. The algorithm can't
# deal with this.
# i = 4 is the special case where 2 is one of the primes. The algorithm
# skips 2 as an optimization.
m[0] = 0
m[1] = 1

for i in range(4, n+1, 2):
    debug(f'considering {i}')

    # keep a and b valid
    if p[a+1] <= i//2:
        a += 1
    if p[b+1] <= i-2:
        b += 1
    debug(f'  largest prime at most i/2 {p[a]}')
    debug(f'  largest prime at most i-2 {p[b]}')

    c = b # p[c] is the next prime to look out for.

    for j in range(1, a+1): # The 2nd half and the prime 2 are skipped.
        debug(f'    checking {p[j]}')
        # The trial is i-p[j]; we're interested in it equaling p[c].
        t = i-p[j]
        debug(f'    trial {t}')
        # Correct p[c]. This while loop tends to have very few iterations; it's
        # efficient, but inherently unpredictable how many it might need.
        while t < p[c]:
            c -= 1
        debug(f'    looking for {p[c]}')
        if t == p[c]:
            m[(i-2)//2] += 1
            debug(f'    found {i} = {p[j]}+{t}')

if PRINT_MULTIPLICITIES:
    for i in range(2, n+1, 2):
        print(f'{i}: {m[(i-2)//2]}')

plt.plot(range(2, n+1, 2), m)
plt.xlim(2, n)
plt.show()

