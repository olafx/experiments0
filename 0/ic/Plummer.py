'''
Sampling of Plummer model by Plummer (1911).

<filename> <number of objects> <Plummer radius> <distance limit relative to Plummer radius>
'''

import sys
import numpy as np
import h5py

if (len(sys.argv) != 5):
    raise Exception('wrong number of command-line arguments')

filename  = sys.argv[1]
n         = int(sys.argv[2])   # objects
a         = float(sys.argv[3]) # Plummer radius
r_max_rel = float(sys.argv[4]) # distance limit relative to Plummer radius

n_batch = int(np.sqrt(n))

r_max   = a*r_max_rel
v_max   = np.sqrt(2*n/np.sqrt(r_max**2+a**2))
# Plummer pdf max is at v = r = 0
pdf_max = 24*np.sqrt(2)/(7*np.pi**3)*a**-1.5*n**-.5

r_v     = np.empty([2, n])
pos_vel = np.empty([2, n, 3])
rng     = np.random.default_rng(0)

# standard rejection sampling
i = 0
while i < n:
    r = rng.uniform(0, r_max, n_batch)
    v = rng.uniform(0, v_max, n_batch)
    p = rng.uniform(0, pdf_max, n_batch)
    E = .5*v**2-n/np.sqrt(r**2+a**2)
    for j in range(n_batch):
        if i < n and E[j] < 0: # if E > 0, pdf considered 0
            if p[j] < 24*np.sqrt(2)/(7*np.pi**3)*a**2/n**4*(-E[j])**3.5:
                r_v[:,i] = r[j], v[j]
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
