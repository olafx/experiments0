/*
Root finding via the Wijngaarden-Dekker-Brent method. This is a fast, 100%
reliable, difficult to confuse method. It achieves this by being a hybrid
method; it tries to use a superlinear solver (inverse quadratic interpolation),
but switches to a more reliable method (bisection) if the convergence is too
slow. Inverse quadratic interpolation is just the secant method, except inverse
quadratic instead of inverse linear. The secant method's order of convergence is
the golden ratio, around 1.62. The inverse quadratic interpolation method has an
order of convergence around 1.84. Conceptually both methods work by
interpolating the inverse of the function locally as a polynomial, and simply
evaluating that in 0 to calculate the approximate root. Another related method
is Muller's method. Here the function is interpolated instead of its inverse,
and the root of the interpolating polynomial is calculated analytically.

This code is based on zeroin.f from netlib, the classic implementation by Brent.

As a test, a bunch of random trigonometrically modulated 5th order polynomials
are generated. 5th order polynomials have no analytic solutions, and the
modulator makes them non-polynomials, to show that the method works for
arbitrary nonlinear functions. (Of course the roots are just those of the
polynomials combined with those of the modulator, but the solver doesn't know
that.)

f(x) = sin(0.1x) (c5 x^5 + ... + c0)
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>
#include <cfloat>

/*
Sets root if it can find a root and returns true, otherwise returns false. f
must be a simple function of x. dx is the maximum allowed error on the root.
i_max is the maximum allowed number of iterations. [x1, x2] is the considered
range to look for a root in, and the middle of this range is used as an initial
guess.
*/
template <typename A>
bool brent(double &root, const double x1, const double x2, const double dx, const size_t i_max, const A &f)
{
    double a, b, c;
    double d, e;
    double min1, min2;
    double p, q, r, s;
    double dx1, xm;
    double fa, fb, fc;
    a = x1; b = x2; c = x2;
    fa = f(a); fb = f(b);
    // A lot of people include this step. The idea is that this algorithm can be
    // 100% reliable with it. But if we already know 100% there's a root in it,
    // this really doesn't have to be included. It in fact limits the usecase
    // arbitrarily.
    // if (fa * fb > 0)
    //     return false;
    fc = fb;
    for (size_t i = 0; i < i_max; i++)
    {   if (fb * fc > 0)
        {   c = a;
            fc = fa;
            d = b-a;
            e = d;
        }
        if (abs(fc) < abs(fb))
        {   a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // This part is a bit sketchy, there's a potential mistake. DBL_EPSILON
        // is really double the standard machine epsilon, i.e. the variation on
        // the definition. The original Fortran codes uses 2*D1MACH(4), where
        // D1MACH(4) from the Bell labs D1MACH routine is 2^(1-53), i.e. also
        // the variational definition. So this should be correct but I can't
        // verify.
        dx1 = 2*DBL_EPSILON*abs(b)+.5*dx;
        xm = .5*(c-b);
        if (abs(xm) <= dx1 || fb == 0)
        {   root = b;
            return true;
        }
        if (abs(e) >= dx1 && abs(fa) > abs(fb))
        {   s = fb/fa;
            if (a == c)
            {   p = 2*xm*s;
                q = 1-s;
            }
            else
            {   q = fa/fc;
                r = fb/fc;
                p = s*(2*xm*q*(q-r)-(b-a)*(r-1));
                q = (q-1)*(r-1)*(s-1);
            }
            p > 0 ? q = -q : p = -p;
            min1 = 3*xm*q-abs(dx1*q);
            min2 = abs(e*q);
            if (2*p < std::min(min1, min2))
            {   e = d;
                d = p/q;
            }
            else
            {   d = xm;
                e = d;
            }
        }
        else
        {   d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (abs(d) > dx1)
            b += d;
        else
            b += xm < 0 ? -abs(dx1) : abs(dx1);
        fb = f(b);
    }
    return false;
}

/*
Evaluate the function f, as discussed above. The coefficients are given as an
array in c. The polynomial order of g is templated.
*/
template <size_t n>
double f(const double *const c, const double x)
{
    double t = c[n];
    for (size_t i = 0; i < n; i++)
        t = t*x+c[n-1-i];
    return sin(.1*x)*t;
}

int main()
{
    constexpr size_t n = 5;   // polynomial order
    constexpr size_t N = 1e3; // number of polynomials
    constexpr double c_range[2] {-10, 10}; // coefficient range
    constexpr double x_range[2] {-1, 10};  // range to look for roots in
    constexpr double eps = .5*DBL_EPSILON; // I don't use the variational definition
    // The relative error should be at least as good as machine epsilon.
    constexpr double dx = eps*std::max(x_range[0] > 0 ? x_range[0] : -x_range[0],
                                       x_range[1] > 0 ? x_range[1] : -x_range[1]); // abs isn't declared constexpr in the standard
    constexpr size_t i_max = 128; // maximum allowed number of iterations

    // RNG setup. Choosing a runtime seed here, why not. Using a 64 bit Mersenne
    // twister; perfect for doubles, and easily available.
    std::random_device seeder;
    std::mt19937_64 rng {seeder()};
    std::uniform_real_distribution<double> distribution {c_range[0], c_range[1]};

    // Generate some coefficients. Let's just store them together.
    auto *c = new double[(n+1)*N];
    for (size_t i = 0; i < (n+1)*N; i++)
        c[i] = distribution(rng);
    // An array for the roots, and an array to remember whether a root was found
    // successfully.
    auto *roots = new double[N];
    auto *success = new bool[N];

    // Let's evaluate the roots. See how many it finds successfully. Should be
    // all obviously, the roots exist and this method is 100% reliable.
    for (size_t i = 0; i < N; i++)
    {
        auto f_simple = [&](const double x) { return f<n>(c+(n+1)*i, x); };
        success[i] = brent(roots[i], x_range[0], x_range[1], dx, i_max, f_simple);
    }
    size_t success_count = 0;
    for (size_t i = 0; i < N; i++)
        if (success[i])
            success_count++;

    // Print some of the results. Would like to see how close the roots are.
    printf("%zu/%zu successful\n\n", success_count, N);
    for (size_t i = 0; i < N; i++)
    {   printf("polynomial %zu/%zu\n", i+1, N);
        if (success[i])
            printf("  f(%+le) = %+le\n", roots[i], f<n>(c+(n+1)*i, roots[i]));
        else
            printf("  failure\n");
    }

    delete[] roots;
    delete[] success;
}
