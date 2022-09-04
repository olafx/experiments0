# 6

Iteratively solving nonlinear equations:
-   Halley's method
    -   Works on any function, but finds one root.
    -   First and second derivatives are required.
    -   Cubic convergence, so extremely fast, faster than its cousin Newton-Raphson.
    -   More reliable than its cousin Newton-Raphson, but still overall a glass cannon.
-   Wijngaarden-Dekker-Brent method
    -   Works on any function, but finds one root.
    -   100% reliable.
    -   Superlinear (which is amazing for a 100% reliable method) but subquadratic.
    -   A complicated hybrid method.
-   Companion matrix eigenvalue
    -   Polynomials only, but finds all roots (trivially).
    -   Eigenvalues are evaluated via the QR algorithm.
        -   The QR decomposition is done via Householder reflection.
    -   A degree n polynomial has a nxn companion matrix, so this method has O(n^2) memory complexity, and thereby questionable scaling to large n.
    -   Cubic convergence.
    -   Complicated, due to the multiple layers to solving the problem.
-   Laguerre's method
    -   Polynomials only, but finds all roots (via deflation).
    -   Very reliable, altho technically not for 100% of polynomials.
    -   Roots are polished via Newton-Raphson. Newton-Raphson is chosen because it'd be a shame not to include it anywhere in a set of nonlinear solving examples.

Everything has been implemented in double precision because that's the standard. These methods will not work in single precision, altho for most of them, the switch to single precision is trivial.

Everything has been implemented as an example; each method is a function in a main file, and the main function runs some more than decent tests. As with everything in this repo, it's all just examples and tests that are applicable to real world problems; this is not a library!
