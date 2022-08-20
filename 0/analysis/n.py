import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

filename = sys.argv[1]

fp = h5py.File(filename, 'r')

n    = fp['pos,vel'].shape[2]
pos  = fp['pos,vel'][:,0,:,:]
vel  = fp['pos,vel'][:,1,:,:]
time = fp['time'][:]

fp.close()

# analysis via exact velocity
total_vel  = np.sum(vel, axis=1)
total_vel  = np.sqrt(np.sum(total_vel**2, axis=1))
vel2       = np.sum(vel**2, axis=2)
kin_energy = .5*(np.sum(vel2, axis=1))

print('maximum vel', np.sqrt(np.max(vel2)))

plt.figure('total kinetic energy')
plt.xlabel('time [a.u.]')
plt.ylabel('total kinetic energy [a.u.]')
plt.plot(time, total_kin_energy, 'k')
plt.xlim(time[0], time[-1])

plt.figure('total velocity')
plt.xlabel('time [a.u.]')
plt.ylabel('total velocity [a.u.]')
plt.plot(time, total_vel, 'k')
plt.xlim(time[0], time[-1])

plt.tight_layout()
plt.show()
