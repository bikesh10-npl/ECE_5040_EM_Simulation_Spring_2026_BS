
import numpy as np
import matplotlib.pyplot as plt

nx = 400 
nt = 1500 
c0 = 299792458.0
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7
f_center = 10e9 
f_max = 12e9            
lam0= c0 / f_max 
dz = lam0/ 20    

S = 0.5
dt = S * dz / c0

ce = dt / (eps0 * dz)
ch = dt / (mu0 * dz)

Ez = np.zeros(nx)
Hy = np.zeros(nx + 1)
source_history = []

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(Ez, color='blue', lw=1.5)
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel('Cell Index (i)')
ax.set_ylabel('Electric Field (Ez)')
ax.set_title("1D FDTD Simulation: 10GHz Pulse hitting PEC")

for tn in range(nt):

   #H field update
    Hy[1:-1] += ch * (Ez[1:] - Ez[:-1])

#E field update
    Ez[1:-1] += ce * (Hy[2:-1] - Hy[1:-2])

    #PEC
    Ez[0] = 0  
    Ez[-1] = 0 
    
    t0 = 80 * dt 
    sigma = 20 * dt 
    pulse = np.exp(-0.5 * ((tn * dt - t0) / sigma)**2) * np.sin(2 * np.pi * f_center * tn * dt) 
    
    Ez[nx // 4] += pulse  
    source_history.append(pulse)

    if tn % 20 == 0:
        line.set_ydata(Ez)
        ax.set_title(f"Time Step: {tn}")
        plt.draw()
        plt.pause(0.001)
     
     #Snapshots   
    if tn == 200:
        ax.set_title(f"Before Reflection at time step {tn}")
        plt.draw()
        plt.savefig("Before_Reflection.pdf", dpi=300)   
    if tn == 280:
        ax.set_title(f"During Reflection at time step {tn}")
        plt.draw()
        plt.savefig("During_Reflection.pdf", dpi=300) 
    if tn == 800:
        ax.set_title(f"After Reflection at time step {tn}")
        plt.draw()
        plt.savefig("After_Reflection.pdf", dpi=300) 

plt.ioff() 
plt.close(fig)

N = len(source_history)
freq_data = np.fft.fft(source_history)
frequencies = np.fft.fftfreq(N, dt)

plt.figure(figsize=(10, 5))
positive = frequencies >= 0
freq_Ghz = frequencies[positive] / 1e9
magnitdue = np.abs(freq_data[positive])
plt.plot(freq_Ghz, magnitdue, color='red', lw=2)

plt.axvspan(8, 12, color='gray', alpha=0.3, label='X-band (8-12 GHz)')

plt.title('Frequency Spectrum (FFT) of the Source Pulse')
plt.xlabel('f (GHz)')
plt.ylabel('Magnitude')
plt.xlim(0, 20)
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


