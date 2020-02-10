import numpy as np

pi = 22/7
def f(x):
    return 3 + (x-3.0)**2.0 + 5.0*np.exp(-x/3.0 + 2.0) - 50.0*np.sin(0.1*x*pi)

#Value of a 
p = input('Lower Bounds')
#Value of b
q = input('Upper Bounds')
#Calculation of the length of the given user parameters
L = float(q) - float(p)

n = 0
while(1):
    if L>0.000001:
        x_mean = (float(p) + float(q))/2.0
        L = float(q) - float(p)
        x1 = float(p) + (L/4.0)
        x2 = float(q) - (L/4.0)
        if f(x1) < f(x_mean):
            q = x_mean
            x_mean = x1
        elif f(x2) < f(x_mean):
            p = x_mean
            x_mean = x2
        elif f(x2) >= f(x_mean):
            p = x1
            q = x2
            n = n+1
        print(L)
    else:
        break
x_minimum = x_mean

