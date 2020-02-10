import numpy as np
import operator
import collections

pi = 22/7
def f(x):
    return 3 + (x-3.0)**2.0 + 5.0*np.exp(-x/3.0 + 2.0) - 50.0*np.sin(0.1*x*pi)

#x1 = input('Enter the initial value of x i.e x1')
x1 = -10
#Step size for estimation
step = 2

#Enter the upper bounds
b = input('Enter the maximum value')
#Enter the lower bounds
a = input('Enter the minimum value')
#Enter the number of iterations for a given quadratic estimation
n = input('Enter the number of iterations')

#Evaluate the given function at x1
fx1 = f(x1)
#X2
x2 = x1 + step
#Evaluate the given function at x2
fx2 = f(x2)

#Now, to check where the algorithm is heading on the number scale
if f(x1)>f(x2):
    #Program will head towards positive step size progression
    step1 = step
elif f(x2)>f(x1):
    #Program will head toward a negative step size progression
    step1 = -step

i = 1
#We'll initialize the loop to begin with iterative process for the quadratic estimation method
#Definining the new system of x1,x2 and x3 for the iterative loop cycle
while i <= int(n):
    lx1 = x1
    lx2 = lx1 + step
    lx3 = lx1 + 2*step
    flx1 = f(lx1)
    flx2 = f(lx2)
    flx3 = f(lx3)
    a0 = flx1
    a1 = (flx2 - flx1)/(lx2 - lx1)
    a2 = (((flx3 - flx1)/(lx3 - lx1))-a1)/(lx3 - lx2)
    x_bar = (((lx1 + lx2)/2)-(a1/2*a2))
    fx_bar = f(x_bar)
    #Now input all the elements lx1 , lx2 , lx3 and x_bar into an array to seek for the sorting of the same array
    A = [lx1,lx2,lx3,x_bar]
    B = [flx1,flx2,flx3,fx_bar]
    #Sorting the A array which holds the x value according to the maxima and minima of the f(x) values in B array
    #To Accomplish the same, C array has been generated
    C = {B[i]:A[i] for i in range(len(A))}
    sorted_C = sorted(C.items(), key=lambda kv:kv[0])
    #sorted_C = collections.OrderedDict(sorted_C)
    x1 = sorted_C[0][1]
    i = i + 1
    break