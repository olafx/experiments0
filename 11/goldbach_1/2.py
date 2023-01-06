'''
An algorithm to calculate the number of ways i can be represented as the sum
of 2 primes for a range of even integers i. Goldbach's conjecture says there
should be at least one way for every i > 3.

This algorithm is an advanced version of the simple solution presented by
dynamic programming: simply add all prime pairs and add to the counter of
whatever the sum comes out to be. The issue with the basic algorithm is that
the sum can be all over the place, and represents an address. Ideally, we would
sum over the primes in an order so that the sum is constant, minimizing the
jumping around in memory. This can be achieved by accounting for known prime
number distributions and following contour lines of the two dimensional function
r(a)+r(b), where r(a) and r(b) are functions which estimate the size of the a-th
and b-th primes respectively, e.g. r(a) = a ln a.

This algorithm is not intended to be very practical, but it's an interesting
idea.
'''

