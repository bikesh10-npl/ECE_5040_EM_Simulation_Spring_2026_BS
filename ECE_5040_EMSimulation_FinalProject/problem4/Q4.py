import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

nx = 1200
nt = 8000

c0 = 299792458.0
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7

f0 = 192e12
lam0 = c0 / f0

n_air = 1.0
n_si = 3.48

dx = lam0 / (n_si * 25)
S = 0.5
dt = S * dx / c0

interface_idx = nx // 2

er = np.ones(nx) * (n_air**2)
er[interface_idx:] = n_si**2

ce = dt / (eps0 * er * dx)
ch = dt / (mu0 * dx)

source_pos = nx // 4
refl_probe = interface_idx - 150
trans_probe = interface_idx + 150



er_ref = np.ones(nx) * (n_air**2)
ce_ref = dt / (eps0 * er_ref * dx)

Ez = np.zeros(nx)
Hy = np.zeros(nx + 1)

inc_record = []

for tn in range(nt):

    # H-update
    Hy[1:-1] += ch * (Ez[1:] - Ez[:-1])

    ez_low_old = Ez[1]
    ez_high_old = Ez[-2]

    # E-update
    Ez[:] += ce_ref * (Hy[1:] - Hy[:-1])

    # Simple ABC
    Ez[0] = ez_low_old
    Ez[-1] = ez_high_old

    t0, sigma = 150 * dt, 30 * dt

    pulse = np.exp(
        -0.5 * ((tn * dt - t0) / sigma) ** 2
        ) * np.sin(2 * np.pi * f0 * tn * dt)

    Ez[source_pos] += pulse

    inc_record.append(Ez[refl_probe])


Ez = np.zeros(nx)
Hy = np.zeros(nx + 1)

total_record = []
trans_record = []

for tn in range(nt):

    # H field update
    Hy[1:-1] += ch * (Ez[1:] - Ez[:-1])

    ez_low_old = Ez[1]
    ez_high_old = Ez[-2]

    # E field update
    Ez[:] += ce * (Hy[1:] - Hy[:-1])

    # ABC
    Ez[0] = ez_low_old
    Ez[-1] = ez_high_old

    
    t0, sigma = 150 * dt, 30 * dt

    pulse = np.exp(
        -0.5 * ((tn * dt - t0) / sigma) ** 2
        ) * np.sin(2 * np.pi * f0 * tn * dt)

    Ez[source_pos] += pulse

    total_record.append(Ez[refl_probe])
    trans_record.append(Ez[trans_probe])



refl_record = np.array(total_record) - np.array(inc_record)



freqs = fftfreq(nt, dt)
mask = (freqs >= 180e12) & (freqs <= 200e12)
f_thz = freqs[mask] / 1e12

FFT_inc = fft(inc_record)[mask]
FFT_refl = fft(refl_record)[mask]
FFT_trans = fft(trans_record)[mask]

eps_small = 1e-15

R_numerical = (np.abs(FFT_refl) / np.abs(FFT_inc)) ** 2
T_numerical = (n_si / n_air) * (np.abs(FFT_trans) / np.abs(FFT_inc)) ** 2

R_analytical = ((n_air - n_si) / (n_air + n_si)) ** 2

T_analytical = 1 - R_analytical


plt.figure(figsize=(10, 4))
plt.plot(er, 'b', label=r'$\epsilon_r$')
ez_scaled = Ez / np.max(np.abs(Ez)) 
plt.plot(ez_scaled + 5, 'r', label='Ez Snapshot')
plt.axvline(interface_idx, color='k', linestyle='--')
plt.title("1. Dielectric Profile and E field")
plt.xlabel("Grid x")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)



plt.figure(figsize=(10, 4))
plt.plot(refl_record, color='orange')
plt.title("2. Time-domain Signal of reflected pulse")
plt.xlabel("Time Step")
plt.ylabel("Ez")
plt.grid(True)



plt.figure(figsize=(10, 4))
plt.plot(f_thz, R_numerical, 'b-', label='Numerical R')
plt.axhline(R_analytical, color='k', ls='--', label='Analytical R')
plt.title("3. Reflection Coefficient (R) vs Frequency")
plt.xlabel("Frequency (THz)")
plt.ylabel("R")
plt.legend()
plt.grid(True)


plt.figure(figsize=(10, 4))
plt.plot(f_thz, T_numerical, 'r-', label='Numerical T')
plt.axhline(T_analytical, color='k', ls='--', label='Analytical T')
plt.title("4. Transmission Coefficient (T) vs Frequency")
plt.xlabel("Frequency (THz)")
plt.ylabel("T")
plt.legend()
plt.grid(True)

plt.show()