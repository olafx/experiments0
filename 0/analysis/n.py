'''
Analysis of n-body solution.

<filename>
'''

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

SAVE_FIGS                 = True
PLOT_TOTAL_VELOCITY       = True
PLOT_TOTAL_KINETIC_ENERGY = True
PLOT_X_DENSITY            = True
PLOT_X_DENSITY_RANGE      = (-2, 2)
PLOT_TRAJECTORIES         = True
PLOT_TRAJECTORY_COUNT     = 10

def save_fig(name):
    plt.savefig(os.path.splitext(filename)[0]+'_'+name, format='png', dpi=400)

PLOT_ANYTHING = PLOT_TOTAL_VELOCITY or PLOT_TOTAL_KINETIC_ENERGY or PLOT_X_DENSITY or PLOT_TRAJECTORIES

filename = sys.argv[1]

fp = h5py.File(str(filename), 'r')

n    = fp['pos,vel'].shape[2]
pos  = fp['pos,vel'][:,0,:,:]
vel  = fp['pos,vel'][:,1,:,:]
time = fp['time'][...]

fp.close()

total_vel        = np.linalg.norm(np.sum(vel, axis=1), axis=1)
vel2             = np.sum(vel**2, axis=2)
total_kin_energy = .5*(np.sum(vel2, axis=1))

plt.rcParams['text.usetex'] = True

if PLOT_TOTAL_VELOCITY:
    plt.figure('total velocity')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('total velocity $v$ [a.u.]')
    plt.plot(time, total_vel, 'k')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('total_velocity')

if PLOT_TOTAL_KINETIC_ENERGY:
    plt.figure('total kinetic energy')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('total kinetic energy $T$ [a.u.]')
    plt.plot(time, total_kin_energy, 'k')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('total_kinetic_energy')

if PLOT_X_DENSITY:
    img = np.empty((len(time), len(time)))
    for i_t in range(len(time)):
        img[i_t,:] = np.histogram(pos[i_t,:,0], bins=len(time), range=PLOT_X_DENSITY_RANGE)[0]
    plt.figure('x density')
    plt.xlabel('position $x$ [a.u.]')
    plt.ylabel('time $t$ [a.u.]')
    extent = (PLOT_X_DENSITY_RANGE[0], PLOT_X_DENSITY_RANGE[1], time[-1], time[0])
    aspect = (PLOT_X_DENSITY_RANGE[1]-PLOT_X_DENSITY_RANGE[0])/(time[-1]-time[0])
    plt.imshow(img, aspect=aspect, extent=extent, cmap='inferno')
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig('x_density')

if PLOT_TRAJECTORIES:
    plt.figure(f'trajectories')
    plt.xlabel('time $t$ [a.u.]')
    plt.ylabel('distance $d$ from $\mathbf{0}$ [a.u.]')
    for j in range(PLOT_TRAJECTORY_COUNT):
        plt.plot(time, np.linalg.norm(pos[:,j,:], axis=1))
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS:
        save_fig(f'trajectories')

if PLOT_ANYTHING and not SAVE_FIGS:
    plt.show()
