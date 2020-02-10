import numpy as np
from scipy.optimize import minimize_scalar

pi = 22/7
def f(x):
    return 3 + (x-3.0)**2.0 + 5.0*np.exp(-x/3.0 + 2.0) - 50.0*np.sin(0.1*x*pi)

x = -10
for i in range(1,10000):
    x1 = x + 2*i
    x2 = x - 2*i
    if f(x)*f(x1) < 0:
        print('Root found at x1', i+1)
    if f(x)*f(x2) < 0:
        print('Root found at x2', i+1)