import pandas as pd
import numpy as np
from scipy.fft import fftn, ifftn


df = pd.read_csv('/Users/laiyuxuan/Desktop/Integration/C2H6/mapped_coordinates_3D.csv')
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

orbitals_C_1s = [(1, 8.4936, 0.352872), (1, 4.8788, 0.473621), (1, 15.4660, -0.001199)]
orbitals_C_2s = [(2, 7.0500, 0.210887), (2, 2.2640, 0.000886), (2, 1.4747, 0.000465), (2, 1.1639, -0.000119)]
orbitals_C_2p = [(2, 7.0500, 0.006977), (2, 3.2275, 0.070877), (2, 2.1908, 0.230802),
                 (2, 1.4413, 0.411931), (2, 1.0242, 0.350701)]

def density_C(x,y,z):
    r = np.sqrt(x**2 + y**2 +z**2)
    return (sto_density(r, orbitals_C_1s) +
            sto_density(r, orbitals_C_2s) +
            sto_density(r, orbitals_C_2p))

def rho_promol(x,y,z):
    return (
        density_C(x,y,z-1.4525) +
        density_C(x,y,z+1.4525) +
        density_H(x+1.9225,y,z-2.1864) +
        density_H(x-0.9612,y-1.6671,z-2.1864) +
        density_H(x-0.9612,y+1.6671,z-2.1864) +
        density_H(x-1.9225,y,z+2.1864) +
        density_H(x+0.9612,y+1.6671,z+2.1864) +
        density_H(x+0.9612,y-1.6671,z+2.1864) 
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