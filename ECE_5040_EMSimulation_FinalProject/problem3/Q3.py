
import numpy as np
import matplotlib.pyplot as plt


c0 = 299792458.0
mu0 = 4.0 * np.pi * 1e-7
eps0 = 8.854e-12

f_cutoff = 9.0e9
a = c0 / (2.0 * f_cutoff)  #width

Nx, Ny = 500, 60
dl = a / 60
dt = dl / (c0 * np.sqrt(2.0)) * 0.99
n_steps = 4000

Ch = dt / (mu0 * dl)
Ce = dt / (eps0 * dl)

mur_coeff = (c0 * dt - dl) / (c0 * dt + dl)


Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

left_old = np.zeros(Ny)
right_old = np.zeros(Ny)


iris_x = Nx // 2
iris_gap_start = 20
iris_gap_end = 40

ref_probe_x = 80
tran_probe_x = 300

ref_data = []
tran_data = []
src_data = []


t0 = 100
spread = 40

for n in range(n_steps):

    # H field update
    Hx[:, :-1] -= Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    # E field update
    Ez[1:-1, 1:-1] += Ce * (
        (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) -
        (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2])
    )

    # PEC
    Ez[:, 0] = 0.0
    Ez[:, -1] = 0.0


    Ez[iris_x, 0:iris_gap_start] = 0.0
    Ez[iris_x, iris_gap_end:] = 0.0

    Ez[0, 1:-1] = left_old[1:-1] + mur_coeff * (Ez[1, 1:-1] - Ez[0, 1:-1])
    Ez[-1, 1:-1] = right_old[1:-1] + mur_coeff * (Ez[-2, 1:-1] - Ez[-1, 1:-1])

    left_old[:] = Ez[1, :]
    right_old[:] = Ez[-2, :]

    x = (n - t0) / spread
    pulse = -2.0 * x * np.exp(-x * x)
    y_profile = np.sin(np.pi * np.arange(Ny) / (Ny - 1))
    Ez[10, :] += y_profile * pulse

    ref_data.append(Ez[ref_probe_x, Ny // 2])
    tran_data.append(Ez[tran_probe_x, Ny // 2])
    src_data.append(pulse)


plt.figure(figsize=(10, 4))
plt.imshow(np.abs(Ez).T, origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='|Ez|')
plt.title("2D E field: Diffraction")
plt.xlabel("x (cells)")
plt.ylabel("y (cells)")


plt.figure(figsize=(10, 4))
plt.plot(ref_data, label="Reflection")
plt.plot(tran_data, label="Transmission")
plt.legend()
plt.title("Time Domain Signals")
plt.xlabel("Time Step")
plt.ylabel("Ez")
plt.grid()

ref_fft = np.fft.rfft(ref_data)
tran_fft = np.fft.rfft(tran_data)
src_fft = np.fft.rfft(src_data)



S11 = np.abs(ref_fft / (src_fft))
S21 = np.abs(tran_fft / (src_fft))

freq = np.fft.rfftfreq(len(ref_data), dt)

band = (freq >= 8e9) & (freq <= 12e9)


plt.figure(figsize=(10, 4))
plt.plot(freq[band]/1e9, 20*np.log10(S11[band]))
plt.title("|S11| Return Loss")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.figure(figsize=(10, 4))
plt.plot(freq[band]/1e9, 20*np.log10(S21[band]))
plt.title("|S21| Transmission")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.show()