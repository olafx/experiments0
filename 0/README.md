# 0

Equal mass n-body gravity experiments.

## Storage

HDF5 is used for storage throughout. This means the format is not standardized, and needs explaining.

### Initial Conditions

In the root group is contained the 'pos,vel' dataset, a (2,n,3) tensor. (HDF5 interprets in row-major order.) This contains the positions and velocities of all n objects. The 2 reflects that velocities append positions in memory. The 3 reflects that 3D vector components are stored together.

This particular ordering is chosen since it's the same ordering that is fastest when solving, so no reordering is necessary.

### Solver Output

The solver output is similar to the initial condition input, except the 'pos,vel' dataset is now a (N,2,n,3) tensor, where N is the number of time steps. Moreover, the time dimension is extendible.

A new dataset 'time' is contained in root. This is an extendible array of length N containing the N different times.

## Initial Conditions

The initial conditions are generated using Python scripts.

-   2-body elliptical
    -   in 0-momentum frame
-   Plummer model
    -   via standard rejection sampling
-   King model
    -   via standard rejection sampling

## Analysis

Some rudimentary analysis is done here using Python scripts. The full n-body analysis is limited to what's doable in O(n), so kinetic energy is included, but not potential energy.

-   2-body
    -   min/max distance
    -   max total velocity
    -   max normalized total energy error
    -   (plot) distance
    -   (plot) potential energy
    -   (plot) normalized total energy
-   n-body
    -   max velocity
    -   (plot) kinetic energy
    -   (plot) total velocity

## Solvers

The solvers are written in C++17 with basic OpenMP parallelization. Nothing fancy here, just universal hardware-invariant common sense. This is not a product after all.

The HDF5 C API is used instead of the C++ one because in my opinion the C API is better designed.

The code is kept as simple as can be, just showcasing the techniques. Every solver is self-contained, self-described, and has its own file.

Solver settings are given via arguments. See code for what the arguments mean, because different solvers have different settings.

An n-body gravity solver has 2 basic components: a force evaluator, and an integrator. The integrator can have a constant timestep, or be adaptive. The solvers implemented use the following force evaluators and integrators.

-   force evaluation
    -   direct
        -   O(n^2)
    -   Barnes-Hut by Barnes, Hut (1986)
        -   O(nlogn)
    -   fast multipole method aka FMM by Rokhlin (1983)
        -   O(n)
-   integration
    -   techniques
        -   leapfrog (KD)
            -   2nd order, symplectic, 1 force evaluation
        -   leapfrog (DKD)
            -   2nd order, symplectic, 1 force evaluation
        -   leapfrog (KDK)
            -   2nd order, symplectic, 2 force evaluations
        -   Forest-Ruth aka FR by Forest, Ruth (1990)
            -   4th order, symplectic, 3 force evaluations
        -   Position Extended Forest-Ruth Like aka PEFRL by Omelyan, Mryglod, Folk (2008)
            -   4th order, symplectic, 4 force evaluations
        -   Yoshida by Yoshida (1990)
            -   6th order, symplectic, 7 force evaluations
            -   8th order, symplectic, 15 force evaluations
    -   adaptivity
        -   techniques
            -   implicit time-symmetrized by Hut, Makino, McMillan (1995)
            -   a technique by Quinn, Katz, Stadel, Lake (1997)
        -   mechanisms
            -   dynamical time via enclosed density
