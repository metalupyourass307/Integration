import numpy as np
from Phi import *

def y_bar_xy(x, y, x_alpha, y_alpha, lambda_y):
    return np.sum(np.log(y-y_alpha + np.sqrt(lambda_y**2 + (x-x_alpha)**2+(y-y_alpha)**2)))
def dy_bar(x, y, y_alpha, x_alpha, lambda_y):
    delta_x = x - x_alpha
    delta_y = y - y_alpha
    sqrt_term = np.sqrt(lambda_y**2 + delta_x**2 + delta_y**2)
    numerator = 1 + (delta_y / sqrt_term)
    denominator = delta_y + sqrt_term
    return np.sum(numerator / denominator)

def alpha_y(x, y_h, y_l, x_alpha, y_alpha, lambda_y):
    y_bar_h = y_bar_xy(x, y_h, x_alpha, y_alpha, lambda_y)
    y_bar_l = y_bar_xy(x, y_l, x_alpha, y_alpha, lambda_y)
    alpha_y = (y_bar_h - y_bar_l) / 4
    return alpha_y
def y_bar_0(x, y_h, y_l, x_alpha, y_alpha, lambda_y):
    y_bar_h = y_bar_xy(x, y_h, x_alpha, y_alpha, lambda_y)
    y_bar_l = y_bar_xy(x, y_l, x_alpha, y_alpha, lambda_y)
    y_bar_0 = (y_bar_h + y_bar_l) / 2
    return y_bar_0
def y_bar(v, x, y_h, y_l, x_alpha, y_alpha, lambda_y):
    alpha_y_u = alpha_y(x, y_h, y_l, x_alpha, y_alpha, lambda_y)
    y_bar_0_u = y_bar_0(x, y_h, y_l, x_alpha, y_alpha, lambda_y)
    return alpha_y_u * Phi(v) + y_bar_0_u