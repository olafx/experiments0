'''
Analysis of 2-body solution.

<filename>
'''

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

SAVE_FIGS         = True
PLOT_ENERGY       = True
PLOT_TOTAL_ENERGY = True
PLOT_DISTANCE     = True

def save_fig(name: str):
    plt.savefig(os.path.splitext(filename)[0]+'_'+name, format='png', dpi=400)

PLOT_ANYTHING = PLOT_ENERGY or PLOT_TOTAL_VELOCITY or PLOT_DISTANCE

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

total_vel     = np.linalg.norm(np.sum(vel, axis=1), axis=1)
total_vel_min = np.min(total_vel)
total_vel_max = np.max(total_vel)

pot_energy = -1/dist

vel2       = np.sum(vel**2, axis=2)
kin_energy = .5*(np.sum(vel2, axis=1))

energy     = kin_energy+pot_energy
energy_min = np.min(energy)
energy_max = np.max(energy)

print(f'min distance {dist_min:.2e}')
print(f'max distance {dist_max:.2e}')
print(f'min total vel {total_vel_min:.2e}')
print(f'max total vel {total_vel_max:.2e}')
print(f'min total energy {energy_min:.2e}')
print(f'max total energy {energy_max:.2e}')

plt.rcParams['text.usetex'] = True

if PLOT_ENERGY:
    plt.figure('energy')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('energy [a.u.]')
    plt.plot(time, pot_energy, 'r', label='potential energy $V$')
    plt.plot(time, kin_energy, 'b', label='kinetic energy $T$')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('energy')

if PLOT_TOTAL_ENERGY:
    plt.figure('total energy')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('total energy $E$ [a.u.]')
    plt.plot(time, energy, 'k')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('total_energy')

if PLOT_DISTANCE:
    plt.figure('distance')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('distance $d$ [a.u.]')
    plt.plot(time, dist, 'k')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('distance')

if PLOT_ANYTHING and not SAVE_FIGS:
    plt.show()
