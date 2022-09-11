'''
Blackbody spectrum.
'''

import matplotlib.pyplot as plt
import numpy as np

T = 6000
λl = 100
λr = 2000
n = 10**3

hc = 1.23984198e3      # eV nm
c = 299792458e9        # nm s^-1
kB = 8.617333262145e-5 # eV K^-1
σ = 2/15*np.pi**5*kB**4/hc**3*c # ev s^-1 nm^-2 K^-4

λ = np.linspace(λl, λr, n)
B = 2*hc*c/λ**5/(np.exp(hc/(λ*kB*T))-1) # eV s^-1 nm^-3 sr^-1
B /= σ*T**4/np.pi # nm^-1 sr^-1

plt.plot(λ, B, 'k', label=f'{T} K')
plt.legend()
plt.xlim(λl, λr)
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'$F_\lambda / E$ [nm$^{-1}$]')
plt.tight_layout()
plt.savefig('Airy_6.png', dpi=400)
