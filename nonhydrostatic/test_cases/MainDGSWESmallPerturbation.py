"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
import hypy1d.swe.riemann_solvers as eq
from hypy1d.interpolation   import Vandermonde1D
from hypy1d.swe.equation    import EqSWESource
from hypy1d.timestepping    import RungeKuttaDG
from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap
#from hypy1d.flux_divergence import FluxDiv1DDGStrong as FluxDiv1DDG
#from hypy1d.flux_divergence import FluxDiv1DDGWeak as FluxDiv1DDG
from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGStrongSWE as FluxDiv1DDG
#from hypy1d.limiter_wd      import Limiter
from hypy1d.limiter_bjsvhy1 import Limiter
#from hypy1d.limiter_none    import Limiter

def Initial(Grid, DGElmt, grav):

  Q = np.zeros((DGElmt.doflength,2))
  b = np.zeros(DGElmt.doflength)

  hm = 0.5
  a  = 10
  xm = 0.0

  xrel = DGElmt.dofcoordinates-xm
  b = hm * np.exp(-a*xrel**2)

  Q[:,0] = hm - b
  Q[:,1] = 0.0
  Q[Q[:,0]<0.0] = 0.0

  mask = abs(DGElmt.dofcoordinates+2.5) <= 0.5
  Q[mask,0] = Q[mask,0] + 0.0001

  return Q, b


def PlotStep(Q, btopo, Q0, titletext):

  H = Q[:,0] + btopo

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(311)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
    #ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')
    ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')

  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
    #ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')
    ax2.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')

  ax2.set_xlim([xmin,xmax])
  ax2.axis([xmin, xmax, 0.4999, 0.5001])

  ax3 = fig.add_subplot(313)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    #ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax3.set_xlim([xmin,xmax])
  #ax3.axis(mrange)

  plt.draw()


N    = 1      # polynomial interpolation order
m    = 1251   # number of grid nodes
xmin = -5.0   # position of left boundary
xmax =  5.0   # position of right boundary
Tmax = 2.4    # end time
CFL  = 0.90   # CFL number
dt   = 0.0002 # time step size
grav = 9.81   # gravitational constant
wettol = 1e-8 # wet tolerance
pltint = 400  # steps between plots
intpts = 2    # number of interpolation points within one element for visualisation

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

eq   = eq.EqSWERusanov(grav, wettol)
#eq    = eq.EqSWEHLLE(grav, wettol)
src   = EqSWESource(eq, dgel, dgqu)
bv    = [BoundaryValueDGZeroExtrap(gr, dgel, gr.bnodes[0]),
         BoundaryValueDGZeroExtrap(gr, dgel, gr.bnodes[1])]
FD    = FluxDiv1DDG(gr, eq, src, dgel, dgqu, bv, wettol)
Lim   = Limiter(gr, eq, src, dgel, wettol)
RK    = RungeKuttaDG(FD, Lim, 2)

# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
fin     = np.finfo(float)
smax    = int(round(Tmax/dt))

# compute initial condition
Q, btopo = Initial(gr, dgel, grav)
src.update(btopo)

QQ = np.zeros((smax+1,dgel.doflength,2))
QQ[0] = Q

# compute cell mean values
Qm = np.zeros((smax+1,gr.elength,2))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

hrange = [xmin, xmax, 0.499, 0.505]
mrange = [xmin, xmax, -1.0, 1.0]
h2range = [0.37, 0.63, -0.5e-2, 4.0e-2]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, btopo, QQ[0], 'Initial conditions')

plt.show(block=False)
plt.draw()

while not plt.waitforbuttonpress():
  pass

# loop over time
t = 0.0
s = 0
#while t < Tmax-fin.resolution:
while s < smax:
  Qnew = RK.step(Q, t, dt)
  Q = Qnew
  t = t + dt
  s = s + 1
  QQ[s] = Q

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

  # plot data every pltint steps
  if (np.mod(s,pltint)==0):
    PlotStep(Q, btopo, QQ[0], 't = ' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)

  print("step: {0:4d}, time: {1:8.5f}, dt = {2:7.5f}".format(s,t,dt))
