import sys
import numpy as np
import h5py

filename = sys.argv[1]
e = float(sys.argv[2]) # orbital eccentricity
a = float(sys.argv[3]) # orbit semi major radius

pos_vel = np.empty([2, 2, 3])

r = a*(1+e)
v = np.sqrt(2/a*(1-e*e)/(1+e*e)**2)

pos_vel[0,0,:] = -.5*r, 0, 0
pos_vel[0,1,:] =  .5*r, 0, 0
pos_vel[1,0,:] = 0, -.5*v, 0
pos_vel[1,1,:] = 0,  .5*v, 0

fp = h5py.File(sys.argv[1], 'w')
fp.create_dataset("pos,vel", data=pos_vel)
fp.close()
