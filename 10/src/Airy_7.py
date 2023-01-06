'''
2D Airy pattern of blackbody.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import j1
import numpy as np

ENHANCE = True
ENHANCE_POWER = .4

lim_a_sin_θ = 1000
res = 1000
n_λ = 200
λ_l = 360 # min 360
λ_r = 830 # max 830
T = 3000


img = np.zeros([res, res, 3])
λ = np.linspace(λ_l, λ_r, n_λ)
a_sin_θ_r = np.linspace(-lim_a_sin_θ, lim_a_sin_θ, res)
a_sin_θ_x, a_sin_θ_y = np.meshgrid(a_sin_θ_r, a_sin_θ_r)
a_sin_θ = np.sqrt(a_sin_θ_x ** 2 + a_sin_θ_y ** 2)


def g(λ, μ, σ1, σ2):
    return np.exp(-.5*((λ-μ)/(σ1 if λ < μ else σ2))**2)

def λ_to_XYZ(λ):
    return np.array([1.056*g(λ, 599.8, 37.9, 31.0)+0.362*g(λ, 442.0, 16.0, 26.7)-0.065*g(λ, 501.1, 20.4, 26.2),
                     0.821*g(λ, 568.8, 46.9, 40.5)+0.286*g(λ, 530.9, 16.3, 31.1),
                     1.217*g(λ, 437.0, 11.8, 36.0)+0.681*g(λ, 459.0, 26.0, 13.8)])

RGB_to_XYZ = 1/0.17697*np.array([[.49, .31, .2], [.17697, .81240, .01063], [0, .01, .99]])
XYZ_to_RGB = np.linalg.inv(RGB_to_XYZ)


hc = 1.23984198e3      # eV nm
c = 299792458e9        # nm s^-1
kB = 8.617333262145e-5 # eV K^-1
σ = 2/15*np.pi**5*kB**4/hc**3*c # ev s^-1 nm^-2 K^-4

def F_E(λ): # nm^-1
    B = 2*hc*c/λ**5/(np.exp(hc/(λ*kB*T))-1)
    return B/(σ*T**4/np.pi)


for i_λ in range(n_λ):
    k = 2*np.pi/λ[i_λ]
    Iλ = F_E(λ[i_λ])
    I = Iλ*(2*j1(k*a_sin_θ)/(k*a_sin_θ))**2
    if ENHANCE:
        I **= ENHANCE_POWER
    RGB = np.matmul(XYZ_to_RGB, λ_to_XYZ(λ[i_λ]))
    for i in range(3):
        img[:,:,i] += I * RGB[i]


# correction necessary since not every color is representable in RGB
img[img < 0] = 0

# basic scaling
img /= n_λ

# extra scaling necessary due to low brightness
img /= np.max(img)

plt.imshow(img, extent=(-lim_a_sin_θ, lim_a_sin_θ, -lim_a_sin_θ, lim_a_sin_θ))
plt.xlabel('$x$')
plt.ylabel('$y$')
title = r'$\sqrt{x^2+y^2} = a\sin\theta$'f', {T} K, {λ_l}-{λ_r} nm'
if ENHANCE:
    title += ', $I^{'f'{ENHANCE_POWER}''}$'
plt.title(title)
plt.savefig('Airy_7.png', dpi=400)
