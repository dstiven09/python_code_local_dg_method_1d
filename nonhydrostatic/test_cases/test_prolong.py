"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGProlong, DGQuadrature1D
from hypy1d.interpolation   import Vandermonde1D

N    = 1       # polynomial interpolation order
xmin = 0.0     # position of left boundary
xmax = 1.0     # position of right boundary
mc   = 3       # number of grid nodes (coarse grid)
mf   = 17      # number of grid nodes (fine grid)
intpts = 2     # number of interpolation points within one element for visualisation

relt  = DGReferenceElement1D(N)
dgqu  = DGQuadrature1D(relt, N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, mc, periodic=False)
grc   = Grid1D(ndcoo, eltnd, ndels)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, mf, periodic=False)
grf   = Grid1D(ndcoo, eltnd, ndels)
dgelc = DGElement1D(grc, relt)
dgelf = DGElement1D(grf, relt)

intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]

QQc = np.array([1.0, 2.0, 1.0, 2.0])
QQf = np.zeros(dgelf.doflength)

fig = plt.figure(1)
fig.clf()
ax1 = fig.add_subplot(111)
for ielt in range(grc.elength):
  intx = grc.nodecoordinates[ielt]+(intxre+1.0)/2.0*grc.elementwidth[ielt]
  ax1.plot(intx, np.dot(intpsi, QQc[dgelc.elementdofs[ielt]]), 'b')
  ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], QQc[dgelc.elementdofs[ielt]]), 'bo')
plt.draw()

QQf = DGProlong(QQc, relt, dgelc, dgelf)

for ielt in range(grf.elength):
  intx = grf.nodecoordinates[ielt]+(intxre+1.0)/2.0*grf.elementwidth[ielt]
  ax1.plot(intx, np.dot(intpsi, QQf[dgelf.elementdofs[ielt]]), 'r--')
  ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], QQf[dgelf.elementdofs[ielt]]), 'rx')
plt.draw()
