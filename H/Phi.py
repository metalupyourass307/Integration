import numpy as np

def Phi(t):
    log_term = np.log((1 + t) / (1 - t))
    return np.log(log_term + np.sqrt(1 + log_term**2))