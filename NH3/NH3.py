import pandas as pd
import numpy as np
from scipy.fft import fftn, ifftn


df = pd.read_csv('/Users/laiyuxuan/Desktop/Integration/NH3/mapped_coordinates_3D.csv')
x_arr = df['x'].values
y_arr = df['y'].values
z_arr = df['z'].values

def slater_norm(n, zeta):
    return (2 * zeta)**(n + 0.5) / np.sqrt(np.math.factorial(2 * n))

def sto_density(r, terms):
    rho = np.zeros_like(r)
    for n, zeta, c in terms:
        N = slater_norm(n, zeta)
        psi = c * N * r**(n - 1) * np.exp(-zeta * r)
        rho += psi**2
    return rho

def density_H(x, y, z):
    r = np.sqrt(x**2 + y**2 +z**2)
    rho = (1 / np.pi) * np.exp(-2 * r)
    return rho

orbitals_N_1s = [(1, 9.9051, 0.354839), (1, 5.7429, 0.472579), (1, 17.9816, -0.001038)]
orbitals_N_2s = [(2, 8.3087, 0.208492), (2, 2.7611, 0.001687), (2, 1.8223, 0.000206), (2, 1.4191, 0.000064)]
orbitals_N_2p = [(2, 8.3490, 0.006323), (2, 3.8827, 0.082938), (2, 2.5920, 0.260147),
                 (2, 1.6946, 0.418361), (2, 1.1914, 0.308272)]

def density_N(x,y,z):
    r = np.sqrt(x**2 + y**2 +z**2)
    return (sto_density(r, orbitals_N_1s) +
            sto_density(r, orbitals_N_2s) +
            sto_density(r, orbitals_N_2p))

def rho_promol(x,y,z):
    return (
        density_N(x,y,z) +
        density_H(x,y+1.7716,z+0.7208) +
        density_H(x-1.5351,y-0.8855,z+0.7208) +
        density_H(x+1.5351,y-0.8855,z+0.7208)
    )

N = int(round(len(x_arr) ** (1/3)))
x_grid = x_arr.reshape(N, N, N)
y_grid = y_arr.reshape(N, N, N)
z_grid = z_arr.reshape(N, N, N)

rho_exact_grid = rho_promol(x_grid, y_grid, z_grid)
ck = fftn(rho_exact_grid)
rho_recon_grid = ifftn(ck).real

z_vals = z_grid[0, 0, :]
sorted_z = np.argsort(z_vals)
z_sorted = z_vals[sorted_z]
rho_z_sorted = rho_exact_grid[:, :, sorted_z]
integral_z_exact = np.trapz(rho_z_sorted, z_sorted, axis=2)

y_vals = y_grid[0, :, 0]
sorted_y = np.argsort(y_vals)
y_sorted = y_vals[sorted_y]
integral_y_exact = np.trapz(integral_z_exact[:, sorted_y], y_sorted, axis=1)

x_vals = x_grid[:, 0, 0]
sorted_x = np.argsort(x_vals)
x_sorted = x_vals[sorted_x]
I_exact = np.trapz(integral_y_exact[sorted_x], x_sorted)

rho_z_sorted_n = rho_recon_grid[:, :, sorted_z]
integral_z_numerical = np.trapz(rho_z_sorted_n, z_sorted, axis=2)
integral_y_numerical = np.trapz(integral_z_numerical[:, sorted_y], y_sorted, axis=1)
I_numerical = np.trapz(integral_y_numerical[sorted_x], x_sorted)

integration_error = np.abs(I_numerical - I_exact) / np.abs(I_exact)

print(f"I_numerical (3D) = {I_numerical}")
print(f"I_exact (3D) = {I_exact}")
print(f"Integration relative error = {integration_error:e}")