'''
Sampling of King model by King (1966).

The parameter names follow King (1966).

<filename> <number of objects> <interpolation resolution> <k> <j> <V0> <r_max>
<required effective interpolation resolution>
'''

# TODO: Not 100% sure the distribution is right.
# TODO: It's slow.
# TODO: The program should be smart and not require r_max or threshold_steps_to_boundary.

import sys
import numpy as np
from scipy import special, integrate, interpolate
import h5py
import matplotlib.pyplot as plt

if (len(sys.argv) != 9):
    raise Exception('wrong number of command-line arguments')

filename = sys.argv[1]
n  = int(sys.argv[2]) # objects
N  = int(sys.argv[3]) # interpolation resolution
k  = float(sys.argv[4])
j  = float(sys.argv[5])
V0 = float(sys.argv[6])

r_max = float(sys.argv[7]) # should be slightly above r0
threshold_steps_to_boundary = int(sys.argv[8]) # min steps to boundary, guarantees accuracy

r_v     = np.empty([2, n])
pos_vel = np.empty([2, n, 3])
rng     = np.random.default_rng(0)

def rho(V):
    return 0 if V>0 else np.sqrt(np.pi**3)*k/j**3*np.exp(2*j**2*(V0-V))*special.erf(j*np.sqrt(-2*V))-2*np.pi*k*np.sqrt(-2*V)*np.exp(2*j**2*V0)*(1/j**2-4/3*V)

def rhs(r, q):
    a = 4*np.pi*rho(q[0])
    if r != 0:
        a -= 2*q[1]/r
    return [q[1], a]

# solve differential equation
sol = integrate.solve_ivp(rhs, [0,r_max], [V0,0], t_eval=np.linspace(0, r_max, N))
# interpolate the result
V_interp = interpolate.interp1d(sol.t, sol.y[0], assume_sorted=True)

# check if range is large enough
i = np.searchsorted(sol.y[0] >= 0, True)
if i == N:
    raise Exception(f"can't continue since r_max < r0, increase r_max")
if i < threshold_steps_to_boundary:
    raise Exception('steps to boundary too small; reduce r_max and/or increase N')

print(f'effective interpolation resolution {i}')

# calculate boundary r0, is simple due to linear interpolation
r0 = sol.t[i-1]-sol.y[0,i-1]*r_max/N/(sol.y[0,i]-sol.y[0,i-1])

# other ranges needed for rejection sampling
v_max = np.sqrt(-2*V0)
pdf_max = 1*np.e**-2

# standard rejection sampling
i = 0
while i < n:
    r = rng.uniform(0, r0)
    v = rng.uniform(0, v_max)
    p = rng.uniform(0, pdf_max)
    if p < r**2*v**2*np.exp(-2*j**2*(V_interp(r)-V0))*(np.exp(-j**2*v**2)-np.exp(j**2*2*V_interp(r))):
        r_v[:,i] = r, v
        i += 1

# random spatially isentropic spherical angles
r_theta = np.arccos(rng.uniform(-1, 1, n))
v_theta = np.arccos(rng.uniform(-1, 1, n))
r_phi   = rng.uniform(0, 2*np.pi, n)
v_phi   = rng.uniform(0, 2*np.pi, n)

def spherical_to_Cartesian(r, theta, phi):
    return np.array([r*np.sin(theta)*np.cos(phi),
                     r*np.sin(theta)*np.sin(phi),
                     r*np.cos(theta)])

# isentropic spherical to Cartesian conversion
pos_vel[0] = np.transpose(spherical_to_Cartesian(r_v[0], r_theta, r_phi))
pos_vel[1] = np.transpose(spherical_to_Cartesian(r_v[1], v_theta, v_phi))

# bring to 0-momentum frame
for i in range(3):
    pos_vel[1,:,i] -= np.sum(pos_vel[1,:,i])/n

# writing
fp = h5py.File(filename, 'w')
fp.create_dataset('pos,vel', data=pos_vel)
fp.close()
