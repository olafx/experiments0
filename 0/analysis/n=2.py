'''
Analysis of 2-body solution.

<path>
'''

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

SAVE_FIGS         = True
PLOT_ENERGY       = True
PLOT_TOTAL_ENERGY = True
PLOT_DISTANCE     = True

PLOT_ANYTHING = PLOT_ENERGY or PLOT_TOTAL_ENERGY or PLOT_DISTANCE

plt.rcParams['font.family'] = 'CMU'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (5, 3)

path = Path(sys.argv[1])

def save_fig(name): plt.savefig(path.parent/f'{path.stem}_{name}.png', dpi=400)

fp = h5py.File(path, 'r')
n = fp['pos,vel'].shape[2]
pos = fp['pos,vel'][:,0,:,:]
vel = fp['pos,vel'][:,1,:,:]
time = fp['time'][...]
fp.close()

dist = np.linalg.norm(pos[:,0,:]-pos[:,1,:], axis=1)
dist_min = np.min(dist)
dist_max = np.max(dist)
total_vel = np.linalg.norm(np.sum(vel, axis=1), axis=1)
total_vel_min = np.min(total_vel)
total_vel_max = np.max(total_vel)
pot_energy = -1/dist
vel2 = np.sum(vel**2, axis=2)
kin_energy = .5*(np.sum(vel2, axis=1))
energy = kin_energy+pot_energy
energy_min = np.min(energy)
energy_max = np.max(energy)

print(f'min distance {dist_min:.2e}')
print(f'max distance {dist_max:.2e}')
print(f'min total vel {total_vel_min:.2e}')
print(f'max total vel {total_vel_max:.2e}')
print(f'min total energy {energy_min:.2e}')
print(f'max total energy {energy_max:.2e}')

if PLOT_ENERGY:
    plt.figure('energy')
    plt.xlabel('time [a.u.]')
    plt.ylabel('energy [a.u.]')
    plt.plot(time, pot_energy, 'r', label='potential energy')
    plt.plot(time, kin_energy, 'b', label='kinetic energy')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('energy')

if PLOT_TOTAL_ENERGY:
    plt.figure('total energy')
    plt.xlabel('time [a.u.]')
    plt.ylabel('total energy [a.u.]')
    plt.plot(time, energy, 'black')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS: save_fig('total_energy')

if PLOT_DISTANCE:
    plt.figure('distance')
    plt.xlabel('time [a.u.]')
    plt.ylabel('distance [a.u.]')
    plt.plot(time, dist, 'black')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS: save_fig('distance')

if PLOT_ANYTHING and not SAVE_FIGS: plt.show()
