"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D, L2norm
from hypy1d.interpolation   import Vandermonde1D

xmin = 0.0     # position of left boundary
xmax = 1.0     # position of right boundary
m    = 101     # number of grid nodes
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)

print('piecewise linear elements:')
N = 1 # polynomial interpolation order

relt = DGReferenceElement1D(N)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

Q   = np.ones(dgel.doflength)*2.0
L2Q = L2norm(Q, dgel, dgqu)
print("||2.0||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(4.0)))

Q = dgel.dofcoordinates
L2Q = L2norm(Q, dgel, dgqu)
print("|| x ||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/3.0)))

print('\npiecewise quadratic elements:')
N = 2 # polynomial interpolation order

relt = DGReferenceElement1D(N)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

Q   = np.ones(dgel.doflength)*2.0
L2Q = L2norm(Q, dgel, dgqu)
print("||2.0||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(4.0)))

Q = dgel.dofcoordinates
L2Q = L2norm(Q, dgel, dgqu)
print("|| x ||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/3.0)))

Q = dgel.dofcoordinates**2
L2Q = L2norm(Q, dgel, dgqu)
print("||x^2||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/5.0)))

print('\npiecewise cubic elements:')
N = 3 # polynomial interpolation order

relt = DGReferenceElement1D(N)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

Q   = np.ones(dgel.doflength)*2.0
L2Q = L2norm(Q, dgel, dgqu)
print("||2.0||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(4.0)))

Q = dgel.dofcoordinates
L2Q = L2norm(Q, dgel, dgqu)
print("|| x ||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/3.0)))

Q = dgel.dofcoordinates**2
L2Q = L2norm(Q, dgel, dgqu)
print("||x^2||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/5.0)))

Q = dgel.dofcoordinates**3
L2Q = L2norm(Q, dgel, dgqu)
print("||x^3||_L2[0,1] = {0:6.4f}, difference to exact norm: {1:+6.4e}".format(L2Q, L2Q-np.sqrt(1.0/7.0)))
