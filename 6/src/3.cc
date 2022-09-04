// Solve polynomial roots by creating a companion matrix to the specified
// polynomial, i.e. a matrix whose characteristic equation is the polynomial,
// and finding its eigenvalues. The roots will be the eigenvalues. The
// eigenvalues are solved here via the QR algorithm. The QR decomposition is
// done via Householer reflection. Overall, for a degree n polynomial, this
// amounts to an iterative solver with O(n^2) time and space complexity, and a
// cubic convergence rate.

