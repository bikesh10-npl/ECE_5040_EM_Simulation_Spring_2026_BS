import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

c0 = 299792458.0
f0 = 192e12
lam0 = c0 / f0
mu0 = 4.0 * np.pi * 1e-7
eps0 = 1.0 / (mu0 * c0**2)
n_core = 3.48
n_clad = 1.44
dx = (lam0 / n_core) / 20 
dy = dx
dt = dx / (c0 * np.sqrt(2.0)) * 0.99
Nx, Ny = 400, 120
n_steps = 2000

y_center = Ny // 2
d_cells_core = 5  
d = d_cells_core * dy

er = np.full((Nx, Ny), n_clad**2)
er[:, y_center - d_cells_core : y_center + d_cells_core] = n_core**2

k0 = 2 * np.pi / lam0

def func(neff, n1, n2, d_val, k_val):
    # Transcendental equation for TE mode
    if neff <= n2 or neff >= n1:
        return 1e6
    bte = (neff**2 - n2**2) / (n1**2 - n2**2)
    V = k_val * 2 * d_val * np.sqrt(n1**2 - n2**2)
    return 2 * np.arctan(np.sqrt(bte / (1 - bte))) - V * np.sqrt(1 - bte)

n_eff_sol = fsolve(func, x0=(n_core + n_clad)/2, args=(n_core, n_clad, d, k0))
n_eff = n_eff_sol[0]

beta = k0 * n_eff
kappa = np.sqrt((n_core * k0)**2 - beta**2)
gamma = np.sqrt(beta**2 - (n_clad * k0)**2)

mode_profile = np.zeros(Ny)
for j in range(Ny):
    y = (j - y_center) * dy
    if abs(y) <= d:
        mode_profile[j] = np.cos(kappa * y)
    else:
        mode_profile[j] = np.cos(kappa * d) * np.exp(-gamma * (abs(y) - d))

Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
Ce = dt / (eps0 * dx * er)
Ch = dt / (mu0 * dx)

for n in range(n_steps):
    Hx[:, :-1] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += Ch * (Ez[1:, :] - Ez[:-1, :])
    Ez[1:-1, 1:-1] += Ce[1:-1, 1:-1] * (
        (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) - (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2])
    )
    Ez[5, :] = mode_profile * np.sin(2 * np.pi * f0 * n * dt)


plt.figure()
plt.plot(mode_profile)
plt.title(f"1. Analytical E field Profile")
plt.xlabel("y")
plt.ylabel("Ez")

plt.figure()
plt.imshow(Ez.T, cmap='RdBu', origin='lower', aspect='auto')
plt.title("2. Ez Snapshot steady state")
plt.xlabel("x")
plt.ylabel("y")

plt.figure()
plt.imshow(Ez.T, cmap='coolwarm', origin='lower', aspect='auto', vmin=-0.1, vmax=0.1)
plt.title("3. Phase Fronts")
plt.xlabel("x")
plt.ylabel("y")

Ez_slice = Ez[Nx//2, :]
Hy_slice = -0.5 * (Hy[Nx//2, :] + Hy[Nx//2 - 1, :])

Sx = Ez_slice * Hy_slice
plt.figure()
plt.plot(Sx)
plt.title("4. Poynting Vector")
plt.xlabel("y")
plt.ylabel("Sx")
plt.show()