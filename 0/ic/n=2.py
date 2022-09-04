# elliptical 2-body initial condition, starting at apoapsis, in 0-momentum frame

import sys
import numpy as np
import h5py

if len(sys.argv) != 4:
    raise Exception('wrong number of command-line arguments')

filename = sys.argv[1]
e = float(sys.argv[2]) # orbital eccentricity
a = float(sys.argv[3]) # orbit semi major radius

pos_vel = np.empty([2, 2, 3])

r = a*(1+e)                         # apapsis
v = np.sqrt(2/a*(1-e*e)/(1+e*e)**2) # apapsis velocity

pos_vel[0,0,:] = -.5*r, 0, 0
pos_vel[0,1,:] =  .5*r, 0, 0
pos_vel[1,0,:] = 0, -.5*v, 0
pos_vel[1,1,:] = 0,  .5*v, 0

# writing
with fp as h5py.File(filename, 'w'):
    fp.create_dataset('pos,vel', data=pos_vel)
