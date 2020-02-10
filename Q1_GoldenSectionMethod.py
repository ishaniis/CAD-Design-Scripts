import numpy as np

pi = 22/7
def f(x):
    return 3 + (x-3)**2 + 5.0*np.exp(-x/3 + 2) - 50*np.sin(0.1*x*pi)

#Value of a 
p = input('Lower Bounds')
#Value of b
q = input('Upper Bounds')
#Evaluating the function at end points 
fp = f(int(p))
fq = f(int(q))
L = float(p) - float(q)
x1 = float(p) + 0.618*(L)
x2 = float(q) - 0.618*(L)
#Evaluating the above function on x1 and x2
fx1 = f(x1)
fx2 = f(x2)

n = 0
while(1):
    if L>0.000001:
        L = float(q) - float(p)
        x1 = float(p) + 0.618*(L)
        x2 = float(q) - 0.618*(L)
        if f(x1) > f(x2):
            p = x1
            L = float(q) - float(p)
            d = 0.618*(L)
            x1 = a + d
            x2 = b - d
        elif f(x2) > f(x1):
            q = x2
            L = float(q) - float(p)
            d = 0.618*(L)
            x1 = a + d
            x2 = b - d
            n = n+1
        print(L)
    else:
        break