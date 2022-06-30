import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

filename = sys.argv[1]

fp = h5py.File(filename, 'r')

n = fp['pos,vel'].shape[2]
pos  = fp['pos,vel'][:,0,:,:]
vel  = fp['pos,vel'][:,1,:,:]
time = fp['time'][:]

fp.close()

# analysis via exact velocity
vel2 = np.sum(vel*vel, axis=2)
kin_energy = .5*(np.sum(vel2, axis=1))

print('maximum vel', np.sqrt(np.max(vel2)))

plt.figure('kinetic energy')
plt.xlabel('time [a.u.]')
plt.ylabel('kinetic energy [a.u.]')
plt.plot(time, kin_energy, 'k')
plt.xlim(time[0], time[-1])

plt.tight_layout()
plt.show()
