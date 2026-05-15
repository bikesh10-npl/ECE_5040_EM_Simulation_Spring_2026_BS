import numpy as np
import matplotlib.pyplot as plt


c0 = 299792458.0
eps0 = 8.854187817e-12
mu0 = 4.0e-7 * np.pi

f0 = 10e9
lam0 = c0 / f0

dx = lam0 / 20.0
dy = dx
dt = 0.99 * dx / (c0 * np.sqrt(2.0))

Nx = 300
Ny = 240
n_steps = 1200

pml = 20
m_grade = 4
kappa_max = 5.0
alpha_max = 0.05
sigma_opt = (m_grade + 1) / (150.0 * np.pi * dx * np.sqrt(mu0 / eps0))
sigma_max = 1.2 * sigma_opt


Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

er = np.ones((Nx, Ny))
pec_mask = np.zeros((Nx, Ny), dtype=bool)

#Slit
wall_x = Nx // 2
slit_width = 16
slit_y1 = Ny // 2 - slit_width // 2
slit_y2 = Ny // 2 + slit_width // 2

pec_mask[wall_x, :] = True
pec_mask[wall_x, slit_y1:slit_y2] = False


source_x = wall_x - 12
source_y = Ny // 2

t0 = 80 * dt
sigma = 20 * dt


Ce = dt / (eps0 * er * dx)
Ch = dt / (mu0 * dx)


be = np.zeros(pml)
ce = np.zeros(pml)
bh = np.zeros(pml)
ch = np.zeros(pml)

for q in range(pml):
    pos_e = (pml - q - 0.5) / pml
    pos_h = (pml - q - 1.0) / pml

    sigma_e = sigma_max * (pos_e ** m_grade)
    sigma_h = sigma_max * (pos_h ** m_grade)

    kappa_e = 1.0 + (kappa_max - 1.0) * (pos_e ** m_grade)
    kappa_h = 1.0 + (kappa_max - 1.0) * (pos_h ** m_grade)

    alpha_e = alpha_max * (1.0 - pos_e ** m_grade)
    alpha_h = alpha_max * (1.0 - pos_h ** m_grade)

    be[q] = np.exp(-(sigma_e / kappa_e + alpha_e) * dt / eps0)
    bh[q] = np.exp(-(sigma_h / kappa_h + alpha_h) * dt / eps0)

    if sigma_e == 0.0:
        ce[q] = 0.0
    else:
        ce[q] = sigma_e * (be[q] - 1.0) / ((sigma_e * kappa_e) + (kappa_e ** 2) * alpha_e)

    if sigma_h == 0.0:
        ch[q] = 0.0
    else:
        ch[q] = sigma_h * (bh[q] - 1.0) / ((sigma_h * kappa_h) + (kappa_h ** 2) * alpha_h)

psi_ez_x_low = np.zeros((pml, Ny))
psi_ez_x_high = np.zeros((pml, Ny))
psi_ez_y_low = np.zeros((Nx, pml))
psi_ez_y_high = np.zeros((Nx, pml))

psi_hy_x_low = np.zeros((pml, Ny))
psi_hy_x_high = np.zeros((pml, Ny))
psi_hx_y_low = np.zeros((Nx, pml))
psi_hx_y_high = np.zeros((Nx, pml))


aperture_line = None
pml_probe = []

far_angles = np.linspace(0.0, np.pi, 181)
far_pattern = np.zeros_like(far_angles)

probe_x = Nx - pml + 2
probe_y = Ny // 2

snapshot_step = 900


for n in range(n_steps):

    # H field update
    Hx[:, :-1] = Hx[:, :-1] - Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Hy[:-1, :] + Ch * (Ez[1:, :] - Ez[:-1, :])

    # CPML for Hx (bottom and top)
    for j in range(pml):
        j_low = j
        j_high = Ny - 2 - j

        d_low = (Ez[:, j_low + 1] - Ez[:, j_low]) / dy
        d_high = (Ez[:, j_high + 1] - Ez[:, j_high]) / dy

        psi_hx_y_low[:, j] = bh[j] * psi_hx_y_low[:, j] + ch[j] * d_low
        psi_hx_y_high[:, j] = bh[j] * psi_hx_y_high[:, j] + ch[j] * d_high

        Hx[:, j_low] = Hx[:, j_low] - dt / mu0 * psi_hx_y_low[:, j]
        Hx[:, j_high] = Hx[:, j_high] - dt / mu0 * psi_hx_y_high[:, j]

    # CPML for Hy (left and right)
    for i in range(pml):
        i_low = i
        i_high = Nx - 2 - i

        d_low = (Ez[i_low + 1, :] - Ez[i_low, :]) / dx
        d_high = (Ez[i_high + 1, :] - Ez[i_high, :]) / dx

        psi_hy_x_low[i, :] = bh[i] * psi_hy_x_low[i, :] + ch[i] * d_low
        psi_hy_x_high[i, :] = bh[i] * psi_hy_x_high[i, :] + ch[i] * d_high

        Hy[i_low, :] = Hy[i_low, :] + dt / mu0 * psi_hy_x_low[i, :]
        Hy[i_high, :] = Hy[i_high, :] + dt / mu0 * psi_hy_x_high[i, :]

    # Update E field
    Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1] + Ce[1:-1, 1:-1] * (
        (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) -
        (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2])
    )

    # CPML for Ez (left and right)
    for i in range(pml):
        i_low = i + 1
        i_high = Nx - 2 - i

        d_low = (Hy[i_low, 1:-1] - Hy[i_low - 1, 1:-1]) / dx
        d_high = (Hy[i_high, 1:-1] - Hy[i_high - 1, 1:-1]) / dx

        psi_ez_x_low[i, 1:-1] = be[i] * psi_ez_x_low[i, 1:-1] + ce[i] * d_low
        psi_ez_x_high[i, 1:-1] = be[i] * psi_ez_x_high[i, 1:-1] + ce[i] * d_high

        Ez[i_low, 1:-1] = Ez[i_low, 1:-1] + dt / eps0 * psi_ez_x_low[i, 1:-1]
        Ez[i_high, 1:-1] = Ez[i_high, 1:-1] + dt / eps0 * psi_ez_x_high[i, 1:-1]

    # CPML for Ez (bottom and top)
    for j in range(pml):
        j_low = j + 1
        j_high = Ny - 2 - j

        d_low = (Hx[1:-1, j_low] - Hx[1:-1, j_low - 1]) / dy
        d_high = (Hx[1:-1, j_high] - Hx[1:-1, j_high - 1]) / dy

        psi_ez_y_low[1:-1, j] = be[j] * psi_ez_y_low[1:-1, j] - ce[j] * d_low
        psi_ez_y_high[1:-1, j] = be[j] * psi_ez_y_high[1:-1, j] - ce[j] * d_high

        Ez[1:-1, j_low] = Ez[1:-1, j_low] + dt / eps0 * psi_ez_y_low[1:-1, j]
        Ez[1:-1, j_high] = Ez[1:-1, j_high] + dt / eps0 * psi_ez_y_high[1:-1, j]


    pulse = np.exp(-0.5 * ((n * dt - t0) / sigma)**2) * np.sin(2 * np.pi * f0 * n * dt)
    Ez[source_x, source_y] = Ez[source_x, source_y] + pulse

    # PEC
    Ez[pec_mask] = 0.0


    pml_probe.append(Ez[probe_x, probe_y])


    if n == snapshot_step:
        aperture_line = np.abs(Ez[wall_x, :]).copy()

        r_samp = min(Nx, Ny) // 2 - pml - 8
        center_x = wall_x
        center_y = Ny // 2

        for k in range(len(far_angles)):
            ang = far_angles[k]
            sx = int(center_x + r_samp * np.cos(ang))
            sy = int(center_y + r_samp * np.sin(ang))

            if 0 <= sx < Nx and 0 <= sy < Ny:
                far_pattern[k] = np.abs(Ez[sx, sy])


Ez_plot = Ez.copy()
Ez_plot[pec_mask] = 0.0

plt.figure(figsize=(8, 6))
plt.imshow(Ez_plot.T, cmap='RdBu', origin='lower')
plt.colorbar(label='Ez')
plt.title('1. 2D Near-field Ez Snapshot from Aperture')
plt.xlabel('x cell')
plt.ylabel('y cell')

wall_y = np.arange(Ny)
plt.plot(np.full(Ny, wall_x), wall_y, 'k-', lw=1)
plt.plot([wall_x, wall_x], [slit_y1, slit_y2], color='yellow', lw=3)


plt.figure(figsize=(8, 4))
plt.plot(np.arange(Ny), aperture_line, color='blue', lw=2)
plt.axvline(slit_y1, color='red', ls='--')
plt.axvline(slit_y2, color='red', ls='--')
plt.title('2. |Ez| Sampled Across Aperture Plane')
plt.xlabel('y cell')
plt.ylabel('|Ez|')
plt.grid(True)


far_pattern = far_pattern / (np.max(far_pattern) + 1e-12)
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='polar')
ax.plot(far_angles, far_pattern, color='green', lw=2)
ax.set_title('3. Uncalibrated Far-field Radiation Pattern')



time_array = np.arange(n_steps) * dt * 1e9
plt.figure(figsize=(8, 4))
plt.plot(time_array, np.abs(pml_probe), color='purple', lw=2)
plt.yscale('log')
plt.title('4. Sampled E field deep inside PML')
plt.xlabel('Time (ns)')
plt.ylabel('|Ez|')
plt.grid(True)

plt.tight_layout()
plt.show()