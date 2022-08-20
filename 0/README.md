# 0

N-body gravity.
Storage, initial conditions, analysis, solvers.
Equal mass only.

## Storage

Positions and velocites are stored.
Global time is stored.
HDF5 is used.

## Initial Conditions

-   2-body elliptical
-   Plummer model
-   King model

## Analysis

-   2-body
    -   min/max distance
    -   max total velocity
    -   max normalized total energy error
    -   (plot) distance
    -   (plot) potential energy
    -   (plot) normalized total energy
-   N-body
    -   maximum velocity
    -   (plot) kinetic energy
    -   (plot) total velocity

## Solvers

-   Force Evaluation
    -   Direct
    -   Barnes-Hut
    -   Fast Multiple Method
-   Integration
    -   Leapfrog (KD)
    -   Leapfrog (KDK)
    -   Leapfrog (DKD)
    -   Forest-Ruth, aka FR
    -   PEFRL
    -   Yoshida 6th order
    -   Yoshida 8th order
    -   Leapfrog (DSKD) (time-adaptive via enclosed density, Quinn-Katz-Stadel-Lake separation method)
    -   Yoshida 6th order (time-adaptive via enclosed density, Hut-Makino-McMillan time-symmetrized method)
