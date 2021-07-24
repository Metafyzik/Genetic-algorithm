import numpy as np
from math import pi

# Domain of a Rastringin function
precision = 2**10
lower_bound = -5.12
upper_bound = 5.12
xy_coordinates = np.linspace(lower_bound,upper_bound, precision)
                                                                               
# Rastingin plot
X,Y = xy_coordinates,xy_coordinates
X, Y = np.meshgrid(X, Y)
A = 10
n = 2
Z =  A*n + (X**2 - A*np.cos(2*pi*X) )  +  (Y**2 -A*np.cos(2*pi*Y)) 

def RastriginFun(x,y,A=10,n=2):
    Z =  A*n + (x**2 - A*np.cos(2*pi*x) )  +  (y**2 -A*np.cos(2*pi*y))
    return Z
