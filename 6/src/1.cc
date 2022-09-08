/*
Root finding via Halley's method with exact first and second derivatives. Like
Newton-Raphson, but much more reliable, and cubic instead of quadratic
convergence. Much more reliable however does not mean that it is overall
reliable, this is practically not reliable enough to blindly use it for
arbitrary problems.

As a test, a bunch of random trigonometrically modulated 5th order polynomials
are generated. 5th order polynomials have no analytic solutions, and the
modulator makes them non-polynomials, to show that the method works for
arbitrary nonlinear functions. (Of course the roots are just those of the
polynomials combined with those of the modulator, but the solver doesn't know
that.)

g(x)   = c5 x^5 + ... + c0
f(x)   = sin(0.1x) g(x)

The first and second derivatives of f are trivial, they depend on those of g:

f'(x)  = 0.1 cos(0.1x) g(x) + sin(0.1x) g'(x),
f''(x) = 0.2 cos(0.1x) g'(x) + sin(0.1x) (g''(x) - 0.01 g(x)).

The polynomials g and their first two derivatives are efficiently evaluated
simultaneously. From g, g', and g'' follow f, f', and f'', evaluated simply as
shown above. The coefficients are uniformly randomly distributed in [-10, 10).
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>
#include <cfloat>

/*
Sets root if it can find a root and returns true, otherwise returns false.
f_df_d2f must be a function with 4 parameters. The first is the input x, the
second is the f lvalue, the third is the f' lvalue, and the fourth is the f''
lvalue. The last 3 parameters exist to return, via lvalue. (It is designed this
way because sometimes the evaluation of f, f', and f'' are easier together; it's
more general to put them all together, and lvalue parameter assignment is a nice
way to return multiple values in a way that is generalizable to languages like C
and Fortran.) dx is the maximum allowed error on the root. i_max is the maximum
allowed number of iterations. [x1, x2] is the considered range to look for a
root in, and the middle of this range is used as an initial guess.
*/
template <typename A>
bool halley(double &root, const double x1, const double x2, const double dx, const size_t i_max, const A &f_df_d2f)
{
    double f, df, d2f;
    root = .5*(x1+x2);
    for (size_t i = 0; i < i_max; i++)
    {   f_df_d2f(root, f, df, d2f); // evaluate
        double jump = f/(df-(.5*f*d2f)/df); // for Newton-Raphson the jump would be f/df
        root -= jump;
        if (root < x1 || root > x2) // jumped outside [x1, x2]
            return false;
        if (abs(jump) < dx) // converged
            return true;
    }
    // too many iterations
    return false;
}

/*
Evaluate a polynomial and its first two derivatives. The result is assigned via
lvalue in g, dg, and d2g respectively. The coefficients are given as an array in
c. The polynomial order is templated.
*/
template <size_t n>
void g_dg_d2g(const double *const c, const double x, double &g, double &dg, double &d2g)
{
    g = c[n];
    dg = 0;
    d2g = 0;
    for (size_t i = 0; i < n; i++)
    {   if (i != 0)
            d2g = d2g*x+dg;
        dg = dg*x+g;
        g = g*x+c[n-1-i];
    }
    d2g *= 2;
}

/*
Evaluate the function f and its first two derivatives, as discussed above. The
result is assigned via lvalue in f, df, and d2f respectively. The coefficients
are given as an array in c. The polynomial order of g is templated.
*/
template <size_t n>
void f_df_d2f(const double *const c, const double x, double &f, double &df, double &d2f)
{
    double g, dg, d2g;
    g_dg_d2g<n>(c, x, g, dg, d2g);
    f   = sin(.1*x)*g;
    df  = .1*cos(.1*x)*g+sin(.1*x)*dg;
    d2f = .2*cos(.1*x)*dg+sin(.1*x)*(d2g-.01*g);
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
    constexpr size_t i_max = 16; // 16 iterations should be more than enough for a cubic method, but need enough in case of slow initial convergence

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

    // Let's evaluate the roots. See how many it finds successfully.
    for (size_t i = 0; i < N; i++)
    {   // The f_df_d2f function is not of the right form for halley, because it
        // needs c. Define a simple lambda so the form is right.
        auto f_df_d2f_simple = [&](const double x, double &f, double &df, double &d2f)
        {   f_df_d2f<n>(c+(n+1)*i, x, f, df, d2f);
        };
        success[i] = halley(roots[i], x_range[0], x_range[1], dx, i_max, f_df_d2f_simple);
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
        {   double f, df, d2f;
            f_df_d2f<n>(c+(n+1)*i, roots[i], f, df, d2f);
            printf("  f(%+le) = %+le\n", roots[i], f);
        }
        else
            printf("  failure\n");
    }
}
