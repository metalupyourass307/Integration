import numpy as np
from Phi import *

def x_bar_x(x, x_alpha, lambda_x):
    return np.sum(np.log(x-x_alpha + np.sqrt(lambda_x**2 + (x-x_alpha)**2)))
def dx_bar(x, x_alpha, lambda_x):
    return np.sum(1 / (x - x_alpha + np.sqrt(lambda_x**2 + (x - x_alpha)**2)) *(1 + (x - x_alpha) / np.sqrt(lambda_x**2 + (x - x_alpha)**2)))

def alpha_bar_x(x_alpha, lambda_x, x_h, x_l):
    return (x_bar_x(x_h, x_alpha, lambda_x) - x_bar_x(x_l, x_alpha, lambda_x)) / 4
def x_0(x_alpha, lambda_x, x_h, x_l):
    return (x_bar_x(x_h, x_alpha, lambda_x) + x_bar_x(x_l, x_alpha, lambda_x)) / 2
def x_bar(u, x_alpha, lambda_x, x_h, x_l):
    a_x = alpha_bar_x(x_alpha, lambda_x, x_h, x_l)
    x0 = x_0(x_alpha, lambda_x, x_h, x_l)
    return a_x * Phi(u) + x0
