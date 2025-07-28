import pandas as pd
import numpy as np
from scipy.fft import fftn, ifftn

# Load mapped 3D Cartesian coordinates from file
df = pd.read_csv('/Users/laiyuxuan/Desktop/Integration/H2O/mapped_coordinates_3D.csv')
x_arr = df['x'].values
y_arr = df['y'].values
z_arr = df['z'].values

# Define normalization factor for Slater-type orbitals (STO)
def slater_norm(n, zeta):
    return (2 * zeta)**(n + 0.5) / np.sqrt(np.math.factorial(2 * n))

# Define STO-based electron density from orbital parameters
def sto_density(r, terms):
    rho = np.zeros_like(r)
    for n, zeta, c in terms:
        N = slater_norm(n, zeta)
        psi = c * N * r**(n - 1) * np.exp(-zeta * r)
        rho += psi**2
    return rho

# Define Hydrogen atomic electron density using analytical expression
def density_H(x, y, z):
    r = np.sqrt(x**2 + y**2 +z**2)
    rho = (1 / np.pi) * np.exp(-2 * r)
    return rho

# STO coefficients for Oxygen atom (extracted from basis set)
orbitals_O_1s = [(1, 11.2970, 0.360063), (1, 6.5966, 0.466625), (1, 20.5019, -0.000918)]
orbitals_O_2s = [(2, 9.5546, 0.208441), (2, 3.2482, 0.002018), (2, 2.1608, 0.000216), (2, 1.6411, 0.000133)]
orbitals_O_2p = [(2, 9.6471, 0.005626), (2, 4.3323, 0.126618), (2, 2.7502, 0.328966),
                 (2, 1.7525, 0.395422), (2, 1.2473, 0.231788)]

# Compute total electron density of Oxygen using 1s, 2s, 2p STO components
def density_O(x,y,z):
    r = np.sqrt(x**2 + y**2 +z**2)
    return (sto_density(r, orbitals_O_1s) +
            sto_density(r, orbitals_O_2s) +
            sto_density(r, orbitals_O_2p))

# Define promolecule density of H2O by summing densities from individual atoms
def rho_promol(x,y,z):
    return (
        density_O(x,y,z-0.2214) +
        density_H(x,y-1.4309,z+0.8865) +
        density_H(x,y+1.4309,z+0.8865)
    )

# Compute grid dimension N assuming cubic grid
N = int(round(len(x_arr) ** (1/3)))

# Reshape flat 1D coordinate arrays into 3D grids
x_grid = x_arr.reshape(N, N, N)
y_grid = y_arr.reshape(N, N, N)
z_grid = z_arr.reshape(N, N, N)

# Evaluate exact promolecule density on the 3D grid
rho_exact_grid = rho_promol(x_grid, y_grid, z_grid)

# Perform 3D FFT and inverse FFT to get interpolated density
ck = fftn(rho_exact_grid)
rho_recon_grid = ifftn(ck).real

# Integration procedure along z, then y, then x to compute total density
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

# Repeat the same integration steps on the reconstructed (FFT-based) density
rho_z_sorted_n = rho_recon_grid[:, :, sorted_z]
integral_z_numerical = np.trapz(rho_z_sorted_n, z_sorted, axis=2)
integral_y_numerical = np.trapz(integral_z_numerical[:, sorted_y], y_sorted, axis=1)
I_numerical = np.trapz(integral_y_numerical[sorted_x], x_sorted)

# Compute relative error between numerical and exact density integration
integration_error = np.abs(I_numerical - I_exact) / np.abs(I_exact)

print(f"I_numerical (3D) = {I_numerical}")
print(f"I_exact (3D) = {I_exact}")
print(f"Integration relative error = {integration_error:e}")