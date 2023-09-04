'''
Analysis of n-body solution.

<path>
'''

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SAVE_FIGS = True
PLOT_TOTAL_VELOCITY = True
PLOT_TOTAL_KINETIC_ENERGY = True
PLOT_X_DENSITY = True
PLOT_X_DENSITY_RANGE = (-2, 2)
PLOT_TRAJECTORIES = True
PLOT_TRAJECTORIES_COUNT = 10
PLOT_DIST_MATRIX = True
PLOT_DIST_MATRIX_MAX_SIZE = 32
PLOT_DIST_MATRIX_FPS = 30

PLOT_ANYTHING = PLOT_TOTAL_VELOCITY or PLOT_TOTAL_KINETIC_ENERGY or PLOT_X_DENSITY or PLOT_TRAJECTORIES

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

total_vel = np.linalg.norm(np.sum(vel, axis=1), axis=1)
vel2 = np.sum(vel**2, axis=2)
total_kin_energy = .5*(np.sum(vel2, axis=1))

if PLOT_TOTAL_VELOCITY:
    plt.figure('total velocity')
    plt.xlabel('time [a.u.]')
    plt.ylabel('total velocity [a.u.]')
    plt.plot(time, total_vel, 'black')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS: save_fig('total_velocity')

if PLOT_TOTAL_KINETIC_ENERGY:
    plt.figure('total kinetic energy')
    plt.xlabel('time [a.u.]')
    plt.ylabel('total kinetic energy [a.u.]')
    plt.plot(time, total_kin_energy, 'black')
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS: save_fig('total_kinetic_energy')

if PLOT_TRAJECTORIES:
    plt.figure('trajectories')
    plt.xlabel('time [a.u.]')
    plt.ylabel('distance from origin [a.u.]')
    for j in range(PLOT_TRAJECTORIES_COUNT):
        plt.plot(time, np.linalg.norm(pos[:,j,:], axis=1))
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.tight_layout()
    if SAVE_FIGS: save_fig('trajectories')

if PLOT_X_DENSITY:
    plt.rcParams['figure.figsize'] = (5, 5)
    size = len(time)
    img = np.empty((len(time), size))
    for i_t in range(len(time)):
        img[i_t,:] = np.histogram(pos[i_t,:,0], bins=size, range=PLOT_X_DENSITY_RANGE)[0]
    plt.figure('x density')
    plt.xlabel('position [a.u.]')
    plt.ylabel('time [a.u.]')
    extent = (PLOT_X_DENSITY_RANGE[0], PLOT_X_DENSITY_RANGE[1], time[-1], time[0])
    aspect = (PLOT_X_DENSITY_RANGE[1]-PLOT_X_DENSITY_RANGE[0])/(time[-1]-time[0])
    plt.imshow(img, aspect=aspect, extent=extent, cmap='inferno', interpolation='none')
    plt.tight_layout()
    if SAVE_FIGS: save_fig('x_density')

if PLOT_DIST_MATRIX:
    skip = max(1, n//PLOT_DIST_MATRIX_MAX_SIZE)
    fig, ax = plt.subplots(figsize=(5, 5))
    def frame(i):
        print(f'{i+1:>{len(str(len(time)))}}/{len(time)}')
        diff = pos[i,::skip,np.newaxis]-pos[i,::skip]
        img = np.linalg.norm(diff, axis=-1)
        img = np.log(1+img)
        ax.set_title(f'time {time[i]:.4e} [a.u.]')
        ax.set_xlabel('particle')
        ax.set_ylabel('particle')
        return ax.imshow(img, cmap='inferno', interpolation='none'),
    ani = animation.FuncAnimation(fig, frame, frames=len(time))
    ani.save(path.parent/f'{path.stem}_distance_matrix.mp4', writer='ffmpeg', fps=PLOT_DIST_MATRIX_FPS, dpi=200)

if PLOT_ANYTHING and not SAVE_FIGS: plt.show()
