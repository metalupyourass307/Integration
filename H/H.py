import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn

# Load 3D mapped Cartesian coordinates from CSV file
df = pd.read_csv('/Users/laiyuxuan/Desktop/Integration/H/mapped_coordinates_3D.csv')
x_arr = df['x'].values
y_arr = df['y'].values
z_arr = df['z'].values

# Define the exact analytical electron density function ρ(r) for Hydrogen
def rho_of_r(x, y, z):
    r = np.sqrt(x**2 + y**2 +z**2)
    rho = (1 / np.pi) * np.exp(-2 * r)
    return rho

# Determine the grid size N such that total number of points = N^3
N = int(round(len(x_arr) ** (1/3)))

# Reshape flat arrays into 3D grids for x, y, z coordinates
x_grid = x_arr.reshape(N, N, N)
y_grid = y_arr.reshape(N, N, N)
z_grid = z_arr.reshape(N, N, N)

# Compute exact 3D electron density on the grid
rho_exact_grid = rho_of_r(x_grid, y_grid, z_grid)

# Perform 3D FFT and inverse FFT to reconstruct density
ck = fftn(rho_exact_grid)  # Forward FFT to get plane-wave coefficients
rho_recon_grid = ifftn(ck).real  # Inverse FFT to reconstruct density (real part only)

# Extract z-values from grid and sort them for integration
z_vals = z_grid[0, 0, :]
sorted_z = np.argsort(z_vals)
z_sorted = z_vals[sorted_z]
rho_z_sorted = rho_exact_grid[:, :, sorted_z]  # Sort ρ along z-axis
# Perform numerical integration along z-axis (axis=2)
integral_z_exact = np.trapz(rho_z_sorted, z_sorted, axis=2)

# Extract y-values and sort for integration
y_vals = y_grid[0, :, 0]
sorted_y = np.argsort(y_vals)
y_sorted = y_vals[sorted_y]
# Integrate over y-axis (axis=1), using sorted values
integral_y_exact = np.trapz(integral_z_exact[:, sorted_y], y_sorted, axis=1)

# Extract x-values and sort for final integration
x_vals = x_grid[:, 0, 0]
sorted_x = np.argsort(x_vals)
x_sorted = x_vals[sorted_x]
# Final integration over x-axis to obtain the exact integral
I_exact = np.trapz(integral_y_exact[sorted_x], x_sorted)

# Repeat integration steps using the reconstructed density from inverse FFT
rho_z_sorted_n = rho_recon_grid[:, :, sorted_z]
integral_z_numerical = np.trapz(rho_z_sorted_n, z_sorted, axis=2)
integral_y_numerical = np.trapz(integral_z_numerical[:, sorted_y], y_sorted, axis=1)
I_numerical = np.trapz(integral_y_numerical[sorted_x], x_sorted)

# Calculate relative integration error between reconstructed and exact result
integration_error = np.abs(I_numerical - I_exact) / np.abs(I_exact)

print(f"I_numerical (3D) = {I_numerical}")
print(f"I_exact (3D) = {I_exact}")
print(f"Integration relative error = {integration_error:e}")