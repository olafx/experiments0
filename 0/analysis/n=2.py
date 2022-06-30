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

# distance
dist     = np.linalg.norm(pos[:,0,:]-pos[:,1,:], axis=1)
dist_min = np.min(dist)
dist_max = np.max(dist)

# analysis via numerical velocity
# dt            = np.diff(time)
# vel_num       = np.diff(pos, axis=0)/dt[:,None,None]
# total_vel_num = np.linalg.norm(np.sum(vel_num, axis=1), axis=1)
# vel2_num      = np.sum(vel_num*vel_num, axis=2)
# energy_num    = .5*(np.sum(vel2_num, axis=1))-1/dist[1:]

# analysis via exact velocity
total_vel  = np.sum(vel, axis=1)
total_vel  = np.sum(total_vel*total_vel, axis=1)
vel2       = np.sum(vel*vel, axis=2)
pot_energy = -1/dist
kin_energy = .5*(np.sum(vel2, axis=1))
energy     = kin_energy+pot_energy
total_vel_max = np.max(total_vel)
energy_min    = np.min(energy)
energy_max    = np.max(energy)
energy_rel = (energy_max-energy)/energy_max
energy_rel = energy

print('min distance', dist_min)
print('max distance', dist_max)
print('max total vel error', total_vel_max)
print('max rel total energy error', (energy_max-energy_min)/energy_max)

plt.figure('distance')
plt.xlabel('time [a.u.]')
plt.ylabel('distance [a.u.]')
plt.plot(time, dist, 'k')
plt.xlim(time[0], time[-1])

plt.figure('total velocity')
plt.xlabel('time [a.u.]')
plt.ylabel('total velocity [a.u.]')
plt.plot(time, total_vel, 'k')
plt.xlim(time[0], time[-1])

plt.figure('potential energy')
plt.xlabel('time [a.u.]')
plt.ylabel('potential energy [a.u.]')
plt.plot(time, pot_energy, 'k')
plt.xlim(time[0], time[-1])

plt.figure('relative energy')
plt.xlabel('time [a.u.]')
plt.ylabel('relative energy')
plt.plot(time, energy_rel, 'k')
plt.xlim(time[0], time[-1])

plt.tight_layout()
plt.show()
