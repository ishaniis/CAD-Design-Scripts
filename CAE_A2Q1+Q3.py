import matplotlib.pyplot as plt
import numpy as np
import gekko
import sklearn
import scipy
from mpl_toolkits.mplot3d import Axes3D

P1 = np.array([3,4,2])
P1_derivative = np.array([4,7,2])
P2 = np.array([6,7,5])
P2_derivative = np.array([-3,-8,5])
P = np.concatenate((P1,P2))
P_derivative = np.concatenate((P1_derivative, P2_derivative))

B_x = np.zeros((4,1))
B_y = np.zeros((4,1))
B_z = np.zeros((4,1))

M = np.array(([1,0,0,0],[0,0,1,0] , [-3,3,-2,-1] , [2,-2,1,1]))

np.put(B_x, [0], P1[0])
np.put(B_x, [1], P2[0])
np.put(B_x, [2], P1_derivative[0])
np.put(B_x, [3], P2_derivative[0])

np.put(B_y, [0], P1[1])
np.put(B_y, [1], P2[1])
np.put(B_y, [2], P1_derivative[1])
np.put(B_y, [3], P2_derivative[1])

np.put(B_z, [0], P1[2])
np.put(B_z, [1], P2[2])
np.put(B_z, [2], P1_derivative[2])
np.put(B_z, [3], P2_derivative[2])

MB1 = np.matmul(M,B_x)
MB2 = np.matmul(M,B_y)
MB3 = np.matmul(M,B_z)
P = np.zeros((100,4))
r = np.linspace(0,1,100)
i = 0
for u in np.linspace(0,1,100):
        P[i][:] = [1, u, (u*u), (u*u*u)]
        i = i + 1

F_x = np.matmul(P,MB1) 
F_y = np.matmul(P,MB2)
F_z = np.matmul(P,MB3)

#Scatter Plot on 2D Surface using three coordinates
plt.scatter(F_x,F_y,F_z)

#Plotting the Curve on 2D surface using 2 coordinates
plt.scatter(F_x,F_y)

#Plotting Curve in 3D
#We need to specify the projection to the IDE that we want to display the same 
#on a 3D Surface
#This is used to generate the 3D space moreover a plane to plot the curve
fig = plt.figure()
Three_dimensional_plot = fig.gca(projection='3d')
#Now, we can plot the same on the plane using 2 coordinates (F_x,F_y)
Three_dimensional_plot.plot(F_x,F_y,label='3D Representation of Curve')
Three_dimensional_plot.legend()
plt.show()

#3D Scatter plot using three coordinates
ax = plt.axes(projection='3d')
ax.scatter3D(F_x,F_y,F_z)


