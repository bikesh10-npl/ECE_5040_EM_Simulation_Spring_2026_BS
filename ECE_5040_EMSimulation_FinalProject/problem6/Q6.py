import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

c0 = 299792458.0
f0 = 190e12  
lam0 = c0 / f0
n_core, n_clad = 3.48, 1.44
dx = (lam0 / n_core) / 20 
dy = dx
dt = dx / (c0 * np.sqrt(2.0)) * 0.99
Nx, Ny = 500, 500 
n_steps = 5000
y_center = 100 
#L Bend 
er = np.full((Nx, Ny), n_clad**2)
mid_x = 250 
er[:mid_x, y_center-5:y_center+5] = n_core**2
er[mid_x-5:mid_x+5, y_center:] = n_core**2

k0 = 2 * np.pi / lam0
n_eff = 2.85 
kappa = np.sqrt((n_core * k0)**2 - (n_eff * k0)**2)
gamma = np.sqrt((n_eff * k0)**2 - (n_clad * k0)**2)
d = 5 * dy
mode_profile = np.zeros(Ny)
for j in range(Ny):
    y_dist = (j - y_center) * dy
    if abs(y_dist) <= d:
        mode_profile[j] = np.cos(kappa * y_dist)
    else:
        mode_profile[j] = np.cos(kappa * d) * np.exp(-gamma * (abs(y_dist) - d))

Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
eps0, mu0 = 8.854e-12, 1.256e-6
Ce = dt / (eps0 * dx * er)
Ch = dt / (mu0 * dx)
mur_coeff = (c0 * dt - dx) / (c0 * dt + dx)

ez_left_old = np.zeros(Ny)
ez_right_old = np.zeros(Ny)
ez_top_old = np.zeros(Nx)
ez_bottom_old = np.zeros(Nx)

energy_density = np.zeros((Nx, Ny))
time_signal_in = []
time_signal_out = []

t0 = 10e-14
sigma = 3e-14

for n in range(n_steps):
    ez_left_old[:] = Ez[1, :]
    ez_right_old[:] = Ez[-2, :]
    ez_top_old[:] = Ez[:, -2]
    ez_bottom_old[:] = Ez[:, 1]  



    Hx[:, :-1] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += Ch * (Ez[1:, :] - Ez[:-1, :])
    Ez[1:-1, 1:-1] += Ce[1:-1, 1:-1] * (
        (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) - (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2])
    )

    #ABC
    Ez[0, :] = ez_left_old + mur_coeff * (Ez[1, :] - Ez[0, :])
    Ez[-1, :] = ez_right_old + mur_coeff * (Ez[-2, :] - Ez[-1, :])
    Ez[:, 0] = ez_bottom_old + mur_coeff * (Ez[:, 1] - Ez[:, 0])
    Ez[:, -1] = ez_top_old + mur_coeff * (Ez[:, -2] - Ez[:, -1]) 

    pulse = np.exp(-0.5 * ((n * dt - t0) / sigma)**2) * np.sin(2 * np.pi * f0 * n * dt)
    Ez[5, :] = pulse * mode_profile

    time_signal_in.append(Ez[20, y_center])
    time_signal_out.append(Ez[mid_x, Ny-50]) 

    energy_density += 0.5 * (eps0 * er * Ez**2 + mu0 * (Hx**2 + Hy**2))


plt.figure(figsize=(8,6))
plt.imshow(er.T, cmap='gray', alpha=0.15, origin='lower') 
plt.imshow(Ez.T, cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
plt.title("1. E-field Pulse at Bend")
plt.xlabel("x")
plt.ylabel("y")

plt.figure()
plt.plot(np.arange(n_steps)*dt*1e15, time_signal_out)  
plt.title("2. Time-domain Signal at TX Probe")
plt.xlabel("Time (fs)")
plt.ylabel("Ez")


freqs = fftfreq(n_steps, dt)
in_fft = np.abs(fft(time_signal_in))
out_fft = np.abs(fft(time_signal_out))
trans_eff = (out_fft / (in_fft + 1e-12))**2 

mask = (freqs >= 180e12) & (freqs <= 200e12)
plt.figure()
plt.plot(freqs[mask]/1e12, trans_eff[mask], color='green', lw=2)
plt.title("3. Transmission Efficiency (180-200 THz)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Efficiency (Pout / Pin)")
plt.grid(True)


plt.figure()
plt.imshow(np.log10(energy_density.T + 1e-15), cmap='magma', origin='lower') 
plt.title("4. Time-Averaged Energy Density in bend region")
plt.colorbar(label="Energy Density")

plt.show()