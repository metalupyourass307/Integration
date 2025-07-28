import numpy as np
from Phi import *



def z_bar_xyz(x, y, z, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z):
    numerator = np.sum(np.log(z - z_alpha + np.sqrt(lambda_z**2 + (x - x_alpha)**2 + (y - y_alpha)**2 + (z - z_alpha)**2)))
    denominator = np.sum(np.sqrt(lambda_x**2 + (x - x_alpha)**2)) * np.sum(np.sqrt(lambda_y**2 + (x - x_alpha)**2 + (y - y_alpha)**2))
    return numerator / denominator

def dz_bar(x, y, z, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z):
    r2 = (x - x_alpha)**2 + (y - y_alpha)**2 + (z - z_alpha)**2
    denom = np.sqrt(lambda_z**2 + r2)
    numerator = np.sum(1 / denom)
    dx = np.sqrt(lambda_x**2 + (x - x_alpha)**2)
    dy = np.sqrt(lambda_y**2 + (x - x_alpha)**2 + (y - y_alpha)**2)
    denominator = np.sum(dx) * np.sum(dy)
    return numerator / denominator

def alpha_z(x, y, z_h, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z):
    z_bar_h = z_bar_xyz(x, y, z_h, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    z_bar_l = z_bar_xyz(x, y, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    alpha_z = (z_bar_h - z_bar_l) / 4
    return alpha_z
def z_bar_0(x, y, z_h, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z):
    z_bar_h = z_bar_xyz(x, y, z_h, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    z_bar_l = z_bar_xyz(x, y, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    z_bar_0 = (z_bar_h + z_bar_l) / 2
    return z_bar_0
def z_bar(w, x, y, z_h, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z):
    alpha_z_u = alpha_z(x, y, z_h, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    z_bar_0_u = z_bar_0(x, y, z_h, z_l, x_alpha, y_alpha, z_alpha, lambda_x, lambda_y, lambda_z)
    return alpha_z_u * Phi(w) + z_bar_0_u