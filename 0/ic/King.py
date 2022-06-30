import sys
import numpy as np
from scipy import special
from scipy import integrate
from scipy import interpolate
import h5py

generator = np.random.default_rng(0)

filename = sys.argv[1]
n = int(sys.argv[2]) # objects
N = int(sys.argv[3]) # interpolation resolution
# King (1966) parameters
k = float(sys.argv[4])
j = float(sys.argv[5])
V0 = float(sys.argv[6])

# TODO this explicit r_max and threshold_steps_to_boundary stuff is bs, do it
#      automatically.
r_max = 10 # should be slightly above r0
threshold_steps_to_boundary = 128 # min steps to boundary, guarantees accuracy

def rho(V):
    return 0 if V>0 else np.sqrt(np.pi**3)*k/j**3*np.exp(2*j**2*(V0-V))*\
                         special.erf(j*np.sqrt(-2*V))-2*np.pi*k*np.sqrt(-2*V)*\
                         np.exp(2*j**2*V0)*(1/j**2-4/3*V)

def rhs(r, q):
    a = 4*np.pi*rho(q[0])
    if r != 0:
        a -= 2*q[1]/r
    return [q[1], a]

# solve differential equation
sol = integrate.solve_ivp(rhs, [0,r_max], [V0,0],
          t_eval=np.linspace(0, r_max, N))
# interpolate the result
V_interp = interpolate.interp1d(sol.t, sol.y[0], assume_sorted=True)

# check if range is large enough
i = 0
while sol.y[0,i] < 0:
    i += 1
    if i == N:
        raise RuntimeError(f"can't continue since r_max < r0")
    print(f'{i} of {N} steps to boundary')
if i < threshold_steps_to_boundary:
    raise RuntimeWarning('steps to boundary too small; reduce r_max and/or '
        'increase N')

# calculate boundary
# TODO there might be a more elegant way to do this with the interpolator
r0 = sol.t[i-1]-sol.y[0,i-1]*r_max/N/(sol.y[0,i]-sol.y[0,i-1])
print(f'boundary at {r0:.2f} of {r_max:.2f}')

# standard rejection sampling
samples_r = np.empty(n)
samples_v = np.empty(n)
i = 0
attempts = 0
while i < n:
    r = generator.uniform(0, r0)
    v = generator.uniform(0, np.sqrt(-2*V_interp(r)))
    # v = generator.uniform(0, np.sqrt(-2*V_interp(r)))
    p = generator.uniform(0, 1/np.e**2)
    if p < r**2*v**2*np.exp(-2*j**2*(V_interp(r)-V0))*\
           (np.exp(-j**2*v**2)-np.exp(j**2*2*V_interp(r))):
        samples_r[i] = r
        samples_v[i] = v
        i += 1
    attempts += 1
print(f'{n/attempts*100:.2f}% sampling efficiency')

# pos and vel isentropic
# TODO I use pos and vel naming scheme, not r and v
samples_r_theta = np.arccos(generator.uniform(-1, 1, n))
samples_v_theta = np.arccos(generator.uniform(-1, 1, n))
samples_r_phi = generator.uniform(0, 2*np.pi, n)
samples_v_phi = generator.uniform(0, 2*np.pi, n)

def spherical_to_Cartesian(r, theta, phi):
    return np.array([r*np.sin(theta)*np.cos(phi),
                     r*np.sin(theta)*np.sin(phi),
                     r*np.cos(theta)])

positions  = spherical_to_Cartesian(samples_r, samples_r_theta, samples_r_phi)
velocities = spherical_to_Cartesian(samples_v, samples_v_theta, samples_v_phi)

# TODO ascontinguousarray is a bullshit fix for VTK trash, not longer necessary
positions  = np.transpose(positions)
velocities = np.transpose(velocities)

# TODO store data
fp = h5py.File(filename, 'w')
pos_vel = np.concatenate((positions, velocities)).reshape(2, n, 3)
fp.create_dataset('pos,vel', data=pos_vel)
fp.close()
