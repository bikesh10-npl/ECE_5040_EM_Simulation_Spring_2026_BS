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

Nx = 420
Ny = 260
n_steps = 1600


pml = 20
m_grade = 4
kappa_max = 5.0
alpha_max = 0.05

sigma_opt = (m_grade + 1.0) / (150.0 * np.pi * dx * np.sqrt(mu0 / eps0))
sigma_max = 1.1 * sigma_opt


Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

er = np.ones((Nx, Ny))
pec_mask = np.zeros((Nx, Ny), dtype=bool)

#Horn Geometry
waveguide_start = 30
flare_start = 120
aperture = 300

guide_half_height = 18
aperture_half_height = 58
center_y = Ny // 2

for i in range(waveguide_start, aperture + 1):

    if i < flare_start:
        top_y = center_y + guide_half_height
        bot_y = center_y - guide_half_height
    else:
        ratio = (i - flare_start) / (aperture - flare_start)
        half_h = guide_half_height + ratio * (aperture_half_height - guide_half_height)
        top_y = int(center_y + half_h)
        bot_y = int(center_y - half_h)

    pec_mask[i, :bot_y] = True
    pec_mask[i, top_y + 1:] = True

# back wall PEC mask
pec_mask[:waveguide_start, :] = True
pec_mask[:waveguide_start, center_y - guide_half_height:center_y + guide_half_height + 1] = False

#source
source_x = waveguide_start + 8
source_y1 = center_y - guide_half_height + 1
source_y2 = center_y + guide_half_height

t0 = 80 * dt
sigma = 20 * dt

Ce = dt / (eps0 * er * dx)
Ch = dt / (mu0 * dx)


be = np.zeros(pml)
ce = np.zeros(pml)
bh = np.zeros(pml)
ch = np.zeros(pml)

for q in range(pml):
    pos_e = (q + 0.5) / pml
    pos_h = (q + 1.0) / pml

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


snapshot_transition = None
snapshot_radiation = None

far_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)
far_pattern = np.zeros_like(far_angles)

probe_x = Nx - 4
probe_y = center_y
probe_pml = []

aperture_phase = None

transition_step = 700
radiation_step = 1200


aperture_data = []


for n in range(n_steps):

    # H field update
    Hx[:, :-1] = Hx[:, :-1] - Ch * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Hy[:-1, :] + Ch * (Ez[1:, :] - Ez[:-1, :])

#CPML Hx
    for j in range(pml):
        j_low = j
        j_high = Ny - 2 - j

        d_low = (Ez[:, j_low + 1] - Ez[:, j_low]) / dy
        d_high = (Ez[:, j_high + 1] - Ez[:, j_high]) / dy

        psi_hx_y_low[:, j] = bh[j] * psi_hx_y_low[:, j] + ch[j] * d_low
        psi_hx_y_high[:, j] = bh[j] * psi_hx_y_high[:, j] + ch[j] * d_high

        Hx[:, j_low] = Hx[:, j_low] - dt / mu0 * psi_hx_y_low[:, j]
        Hx[:, j_high] = Hx[:, j_high] - dt / mu0 * psi_hx_y_high[:, j]

#CPML Hy
    for i in range(pml):
        i_low = i
        i_high = Nx - 2 - i

        d_low = (Ez[i_low + 1, :] - Ez[i_low, :]) / dx
        d_high = (Ez[i_high + 1, :] - Ez[i_high, :]) / dx

        psi_hy_x_low[i, :] = bh[i] * psi_hy_x_low[i, :] + ch[i] * d_low
        psi_hy_x_high[i, :] = bh[i] * psi_hy_x_high[i, :] + ch[i] * d_high

        Hy[i_low, :] = Hy[i_low, :] + dt / mu0 * psi_hy_x_low[i, :]
        Hy[i_high, :] = Hy[i_high, :] + dt / mu0 * psi_hy_x_high[i, :]

#E field update
    Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1] + Ce[1:-1, 1:-1] * (
        (Hy[1:-1, 1:-1] - Hy[0:-2, 1:-1]) -
        (Hx[1:-1, 1:-1] - Hx[1:-1, 0:-2])
    )

#CPML Ez_x
    for i in range(pml):
        i_low = i + 1
        i_high = Nx - 2 - i

        d_low = (Hy[i_low, 1:-1] - Hy[i_low - 1, 1:-1]) / dx
        d_high = (Hy[i_high, 1:-1] - Hy[i_high - 1, 1:-1]) / dx

        psi_ez_x_low[i, 1:-1] = be[i] * psi_ez_x_low[i, 1:-1] + ce[i] * d_low
        psi_ez_x_high[i, 1:-1] = be[i] * psi_ez_x_high[i, 1:-1] + ce[i] * d_high

        Ez[i_low, 1:-1] = Ez[i_low, 1:-1] + dt / eps0 * psi_ez_x_low[i, 1:-1]
        Ez[i_high, 1:-1] = Ez[i_high, 1:-1] + dt / eps0 * psi_ez_x_high[i, 1:-1]


#CPML Ez_y
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

    for j in range(source_y1, source_y2 + 1):
        Ez[source_x, j] = Ez[source_x, j] + pulse

    Ez[pec_mask] = 0.0

    probe_pml.append(Ez[probe_x, probe_y])

    if n == transition_step:
        snapshot_transition = Ez.copy()

    if n == radiation_step:
        snapshot_radiation = Ez.copy()

        radius = 85
        x_center = aperture + 10
        y_center = center_y

        for k in range(len(far_angles)):
            ang = far_angles[k]
            x_pos = int(x_center + radius * np.cos(ang))
            y_pos = int(y_center + radius * np.sin(ang))

            if 0 <= x_pos < Nx and 0 <= y_pos < Ny:
                far_pattern[k] = np.abs(Ez[x_pos, y_pos])

    if n > n_steps - 250:
        aperture_data.append(Ez[aperture - 2, :].copy())


aperture_data = np.array(aperture_data)
Nt_ap = aperture_data.shape[0]
frequency = np.fft.fftfreq(Nt_ap, d=dt)

target_index = np.argmin(np.abs(frequency - f0))
fft_aperture = np.fft.fft(aperture_data, axis=0)

aperture_complex = fft_aperture[target_index, :]
aperture_phase = np.angle(aperture_complex)

opening_top = center_y + aperture_half_height
opening_botm = center_y - aperture_half_height


plt.figure(figsize=(9, 6))
plot1 = snapshot_transition.copy()
plot1[pec_mask] = 0.0
plt.imshow(plot1.T, cmap='RdBu', origin='lower')
plt.colorbar(label='Ez')
plt.title('1. E-field Snapshot showing wave transition into flare')
plt.xlabel('x cell')
plt.ylabel('y cell')


plt.figure(figsize=(9, 6))
plot2 = snapshot_radiation.copy()
plot2[pec_mask] = 0.0
plt.imshow(plot2.T, cmap='RdBu', origin='lower')
plt.colorbar(label='Ez')
plt.title('2. E-field Snapshot from the Aperture')
plt.xlabel('x cell')
plt.ylabel('y cell')

far_pattern = far_pattern / (np.max(far_pattern) + 1e-12)

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='polar')
ax.plot(far_angles, far_pattern, color='green', lw=2)
ax.set_title('3. Normalized Radiation')


y_line = np.arange(Ny)
phase_region = aperture_phase[opening_botm:opening_top + 1]
y_region = y_line[opening_botm:opening_top + 1]

plt.figure(figsize=(8, 4))
plt.plot(y_region, phase_region, color='purple', lw=2)
plt.title('4. Aperture Phase Distribution Across Horn Aperture')
plt.xlabel('y cell')
plt.ylabel('Phase (rad)')
plt.grid(True)

plt.tight_layout()
plt.show()