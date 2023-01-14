import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

filename = sys.argv[1]

fp = h5py.File(filename, 'r')

n    = fp['pos,vel'].shape[2]
pos  = fp['pos,vel'][:,0,:,:]
vel  = fp['pos,vel'][:,1,:,:]
time = fp['time'][...]

fp.close()

total_vel        = np.linalg.norm(np.sum(vel, axis=1), axis=1)
vel2             = np.sum(vel**2, axis=2)
total_kin_energy = .5*(np.sum(vel2, axis=1))

plt.figure('total velocity')
plt.xlabel('time [a.u.]')
plt.ylabel('total velocity [a.u.]')
plt.plot(time, total_vel, 'k')
plt.xlim(time[0], time[-1])
plt.tight_layout()

plt.figure('total kinetic energy')
plt.xlabel('time [a.u.]')
plt.ylabel('total kinetic energy [a.u.]')
plt.plot(time, total_kin_energy, 'k')
plt.xlim(time[0], time[-1])
plt.tight_layout()

# shape of cluster over time
axis = 0
res_horizontal = len(time)
img = np.empty((len(time), res_horizontal))
pos_range = (-3, 3)
for i_t in range(len(time)):
    img[i_t,:] = np.histogram(pos[i_t,:,axis], bins=res_horizontal, range=pos_range)[0]
plt.figure(f'axis {0} density over time')
plt.xlabel('position [a.u.]')
plt.ylabel('time [a.u.]')
plt.imshow(img, extent=(pos_range[0], pos_range[1], time[-1], time[0]), cmap='inferno')
plt.tight_layout()

# distance from 0 over time for a few random objects
i = 10
o = 0
plt.figure(f'trajectory of {i} random objects')
plt.xlabel('time [a.u.]')
plt.ylabel('distance from 0 [a.u.]')
for j in range(i):
    plt.plot(time, np.linalg.norm(pos[:,o+j,:], axis=1))
plt.xlim(time[0], time[-1])
plt.tight_layout()

plt.show()
