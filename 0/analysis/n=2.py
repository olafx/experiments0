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

dist     = np.linalg.norm(pos[:,0,:]-pos[:,1,:], axis=1)
dist_min = np.min(dist)
dist_max = np.max(dist)

total_vel  = np.sum(vel, axis=1)
total_vel  = np.sqrt(np.sum(total_vel**2, axis=1))
vel2       = np.sum(vel**2, axis=2)
pot_energy = -1/dist
kin_energy = .5*(np.sum(vel2, axis=1))
energy     = kin_energy+pot_energy
total_vel_max = np.max(total_vel)
energy_min    = np.min(energy)
energy_max    = np.max(energy)
energy_norm   = (energy_max-energy)/energy_max

print('min distance', dist_min)
print('max distance', dist_max)
print('max total vel', total_vel_max)
print('max norm total energy error', (energy_max-energy_min)/energy_max)

plt.figure('distance')
plt.xlabel('time [a.u.]')
plt.ylabel('distance [a.u.]')
plt.plot(time, dist, 'k')
plt.xlim(time[0], time[-1])

plt.figure('potential energy')
plt.xlabel('time [a.u.]')
plt.ylabel('potential energy [a.u.]')
plt.plot(time, pot_energy, 'k')
plt.xlim(time[0], time[-1])

plt.figure('normalized total energy')
plt.xlabel('time [a.u.]')
plt.ylabel('normalized total energy [a.u.]')
plt.plot(time, energy_norm, 'k')
plt.xlim(time[0], time[-1])

plt.tight_layout()
plt.show()
