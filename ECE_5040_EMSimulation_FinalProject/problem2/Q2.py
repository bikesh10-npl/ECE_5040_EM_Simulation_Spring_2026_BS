
import numpy as np
import matplotlib.pyplot as plt


c0 = 299792458.0
mu0 = 4.0 * np.pi * 1e-7
eps0 = 8.854e-12

f_cutoff = 9.0e9
a = c0 / (2.0 * f_cutoff) 


Nx, Ny = 400, 60
dl = a / 60 
dt = dl / (c0 * np.sqrt(2.0)) * 0.99
n_steps = 4000         


f_source = 7.0e9  # change this for 7 and 11Ghz

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)
mur_coeff = (c0 * dt - dl) / (c0 * dt + dl)

Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
probe_data = []
left_old = np.zeros(Ny)
right_old = np.zeros(Ny)



for n in range(n_steps):
  
    Hx[:, :-1] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    Ez[1:-1, 1:-1] += Ce * ((Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) - 
                            (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2]))

    # PEC
    Ez[:, 0] = 0.0
    Ez[:, -1] = 0.0

    # ABC
    Ez[0, 1:-1] = left_old[1:-1] + mur_coeff * (Ez[1, 1:-1] - Ez[0, 1:-1])
    Ez[-1, 1:-1] = right_old[1:-1] + mur_coeff * (Ez[-2, 1:-1] - Ez[-1, 1:-1])
    left_old[:] = Ez[1, :]
    right_old[:] = Ez[-2, :]

  
    y_profile = np.sin(np.pi * np.arange(Ny) / (Ny - 1))
    
    Ez[5, :] = y_profile * np.sin(2.0 * np.pi * f_source * n * dt) 

    probe_data.append(Ez[250, Ny // 2])


plt.figure(figsize=(10, 4))
plt.imshow(np.abs(Ez).T, cmap='inferno', origin='lower', aspect='auto')
plt.colorbar(label='|Ez| Magnitude')
plt.title(f"Plot {1 if f_source > f_cutoff else 2}: 2D Snapshot ({f_source/1e9} GHz)")
plt.xlabel("x (cells)")
plt.ylabel("y (cells)")

plt.figure(figsize=(10, 4))
plt.plot(probe_data, color='blue')
plt.title(f"Plot 3: Time-Domain Signal at Probe ({f_source/1e9} GHz)")
plt.xlabel("Time Step")
plt.ylabel("Ez Amplitude")
plt.grid(True, ls='--')

if f_source > f_cutoff:
    plt.figure(figsize=(8, 5))
    
    f_vec = np.linspace(9.2e9, 15e9, 100)
    vp_theory = c0 / np.sqrt(1 - (f_cutoff / f_vec)**2)
    plt.plot(f_vec/1e9, vp_theory, 'k-', label='Analytical')

    center_line = np.abs(Ez[:, Ny // 2])
    

    peaks = []
    for i in range(10, Nx - 10): 
        if center_line[i] > center_line[i-1] and center_line[i] > center_line[i+1]:
            peaks.append(i)
            if len(peaks) == 3: 
                break

    measured_lambda_g = (peaks[2] - peaks[1]) * dl
    
    vp_measured = f_source * measured_lambda_g

    plt.scatter(f_source/1e9, vp_measured, color='red', s=100, label='Simulated')
    plt.title("Phase Velocity")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Phase Velocity")  
    plt.legend()
    plt.grid(True)

plt.show()