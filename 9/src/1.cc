/*
Segmented sieve of Eratosthenes.
This algorithm, as implemented, can still be optimized in a few ways.
-   The booleans can be made single bit to save on memory. But this isn't that
    important because while the sieve of Eratosthenes is memory constrained, the
    segmented version is very much time constrained.
-   The boolean array can be made smaller by ignoring multiples of 2, as they're
    not prime anyway. This improves both the space and time required. Maybe this
    optimization can be generalized to multiples of 3 and higher too. But I'm
    not looking into it further.
-   The first step in the sieve can, as an optimization, start from the square.
    This is implemented already, but possibly there exists a similar
    optimization for the consecutive steps. It's not immediately obvious what
    this would be, and I'm not looking into it further.

Because this algorithm is completely self designed and not easy to understand,
everything is explained in detail. It isn't that long and looks simple, but is
pretty difficult to understand due to nontrivial prime number factorization
properies and whatnot.
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>

int main(int argc, char **argv)
{
    // The number of primes in [0, N] is calculated, where N >= 8.
    size_t N;
    if (argc != 2 || sscanf(argv[1], "%zu\n", &N) == 0 || N < 9)
        return EXIT_FAILURE;

    /*
    The problem is split into segments [m-d, m) of length (segment size)
    d = floor(sqrt(n)). The segments are defined via m = d, d+m, d+2m, ...,
    until m is at least N+1, since [..., N+1) = [..., N] includes N.

    This segmentation makes the memory required O(d) = O(sqrt(n)). The primes in
    the first segment must however be remembered also, so the memory required is
    larger than sqrt(n). The prime number theorem guarantees a pi(x) ~ x/ln(x)
    distribution. This however is asymptotic; the real density for small x ~=
    10^8 is more like 1.1x/ln(x). So there is an extra 1.1d/ln(d) required for
    d ~= 10^8, which is absolutely not negligible, yet disappears in big-O, so
    I'm mentioning it.

    d >= 3 is required for the algorithm to work, hence the N >= 8 requirement.
    Why does it fail if d = 2? I have no idea, and don't care to debug it for an
    hour.
    */

    size_t d = sqrt(N); // segment size
    size_t n = 0;       // number of primes thusfar

    /*
    First an array of size d is set up to mark the primes in the segments.
    The other array will contain the primes in [0, d]. Naturally, since the
    number of primes in [0, d] aren't known yet, this isn't allocated yet. These
    two small arrays are the only memory that is required by this algorithm,
    giving it the O(sqrt(n)+sqrt(n)/ln(sqrt(n))) = O(sqrt(n)) memory required,
    as discussed.
    */
    auto *sieve = new bool[d]; // sieve array
    for (size_t i = 0; i < d-1; i++)
        sieve[i] = true;

    /*
    The first segment is [0, d). The first segment is sieved differently from
    the others, so is treated separately. (The last segment will also be sieved
    differently, more on this later.)

    In this segment we will actually include d as well, so check [0, d]. This is
    because we'll need all primes in [0, d] for the next segments. This is an
    exception, in the other segments only [0, d) will be checked. Why these
    primes are needed and why d is included will be explained later in the next
    segments.

    The sieve works by enumerating all multiples of the numbers in [2, sqrt(d)),
    marking each multiple as non-prime. It is trivial that these multiples are
    non-prime, they're constructed by multiplying two numbers, but it is not yet
    trivial that all non-primes will indeed be marked. It will be proven now
    that [2, sqrt(d)) is the smallest, most optimal range to enumerate.

    The start is 2 because the smallest number in the prime factorization of
    any non-prime will be the smallest prime.

    The end is sqrt(d) because any non-prime in [sqrt(d), d) must necessarily
    have a prime factorization in which one number is at most sqrt(d). I.e, from
    all the non-primes a, the one with the most difficult prime factorization
    form is a = sqrt(a) * sqrt(a), in the sense that
    min(sqrt(a), sqrt(a)) = sqrt(a) is the largest minimum prime for any
    possible prime factorization containing any number of primes.
    This is not easy to understand so I will give some examples.
    -   A factorization like a = 2 * (a/2) has a very small minimum prime,
        namely 2, and something like a = cbrt(a) * cbrt(a) * cbrt(a) also has a
        smaller minimum prime because cbrt(a) < sqrt(a) for a > 1. The worst
        case, the last possible minimum prime that must be checked, really is
        sqrt(a). And so it should make sense that after sqrt(d), no new
        non-primes will be marked, so the enumeration process can stop.
    -   Say d = 49, so sqrt(d) = 7. Consider 33, in [sqrt(d), d). According to
        the above, 33 should have already been marked. Indeed, 33 = 3*11, and 3
        was already enumerated since 3 <= sqrt(d). 3 is in fact very small, so
        it was checked a long time ago. What is the most dangerous one, the
        most difficult to find? 36 = 6 * 6. This has equal difficulty as
        42 = 6 * 7. The point is, 6 is the last that needs to be checked here,
        which is sqrt(d)-1. However, sqrt(d) returns a float, and floats are
        floored to integers when casted to integers, so in general it must be
        sqrt(d)-1+1 = sqrt(d).

    An optimization made during the enumeration process of numbers i in
    [2, sqrt(d)) is that we skip right away to i^2. Say a non-prime has a prime
    factorization a = i*j. If j < i, j has already been enumerated, and so a
    has already been marked. If i >= j, a is at least i^2. So we can start at
    i^2.

    Another optimization made during the enumeration process is that only the
    primes in [2, sqrt(d)) are enumerated, not the non-primes. This is done by,
    after each enumeration, finding the next prime as the first unmarked number,
    and using that for the next enumeration. This works because factorizations
    can always be simplified to prime factorizations, so doing all
    possible factorizations is unnecessary. For example, don't bother
    enumerating 4 because 2 has already been enumerated.

    Because 0 and 1 aren't prime, we let sieve[i] depict whether i+2 is prime
    temporarily. This opens up two numbers of free memory. This allows us to fit
    the [0, d] interval of length d+1 in the memory for the general [m-d, m)
    interval of length d.
    */

    for (size_t i = 2; i <= sqrt(d);)
    {   for (size_t e = i*i; e <= d; e += i)
            sieve[e-2] = false;
        for (i++; i <= d && !sieve[i-2]; i++);
    }
    for (size_t i = 2; i < d; i++)
        if (sieve[i-2])
            n++;

    /*
    Now that the number of primes in [0, d] is known, we can store them all.
    They will be used in the next sieves, as will be explained below.
    */
    size_t n1 = n+sieve[d-2];
    auto *prime = new size_t[n1];
    for (size_t i = 2, j = 0; i <= d; i++)
        if (sieve[i-2])
            prime[j++] = i;

    /*
    The next segments [m, m-d) are sieved in much the same way, except instead
    of considering the primes in [0, sqrt(d)), those in [0, d] are considered,
    i.e. the ones we just stored. Any non-prime in [0, N] necessarily has a
    prime factorization such that the smallest of its primes is at most d. It
    should be obvious why by now why this is sufficient; the same principle was
    explained before for the [0, d) interval instead of [0, N]. This is
    fundamentally how a segmented sieve works, it knows that it only needs to
    enumerate [2, d] to get all the primes in [0, N], so it doesn't bother to
    store the ones in (d, N], and achieves O(sqrt(N)) memory this way. Instead
    it just counts how many primes were in [m-d, m), resets the sieve, and moves
    on to [m, m+d) forgetting all the (useless) primes in [m-d, m).

    As discussed, m must be at least N+1 for [0, N] to be counted completely.
    The last segment can be further optimized so isn't done here.

    An optimization made is that in the [m-d, m) interval, only primes up to
    sqrt(m) need to be checked, and m stays smaller than d, so this is useful.

    It is nontrivial where the enumeration should start. In [0, d), the
    enumeration starts at the prime i itself. (But this was optimized to i^2.)
    So the first multiple of the prime is in principle 1 here, which is
    fundamentally because the left side of the interval is 0. In the later
    segments [m-d, d), given some prime p, we know the first multiple will be
    ceil((m-d)/p). This is a good spot to start the enumeration; it's equivalent
    to starting at multiple 1 and skipping right away to the left side of the
    interval. An interesting question now is if there is another optimization
    possible here, a generalization of the i^2 optimization. I think not.
    ceil((m-d)/p) is evaluated using the property ceil(a/b) = floor(a/b)+c,
    where c is 1 if b does not divide a, and otherwise 0.
    */

    size_t m;
    for (m = 2*d; m <= N; m += d)
    {   for (size_t i = 0; i < d; i++)
            sieve[i] = true;
        for (size_t i = 0; i < n1; i++)
        {   size_t p = prime[i];
            if (p > sqrt(m))
                break;
            size_t s = p*((m-d)/p+((m-d)%p != 0));
            for (size_t e = s; e < m; e += p)
                sieve[e-m+d] = false;
        }
        for (size_t i = 0; i < d; i++)
            if (sieve[i])
                n++;
    }

    /*
    The final segment is the same as the other [m-d, m) segments, except for
    some optimizations. The previous optimization of considering primes until
    sqrt(m) instead of until d is no longer useful. The sieving is stopped at N
    instead of m. Not the entire sieve is reset, only until the part that
    corresponds to N.
    */

    for (size_t i = 0; i <= N-m+d; i++)
        sieve[i] = true;
    for (size_t i = 0; i < n1; i++)
    {   size_t p = prime[i];
        size_t s = p*((m-d)/p+((m-d)%p != 0));
        for (size_t e = s; e <= N; e += p)
            sieve[e-m+d] = false;
    }
    for (size_t i = 0; i <= N-m+d; i++)
        if (sieve[i])
            n++;

    printf("%zu\n", n);

    delete[] sieve;
    delete[] prime;
}
