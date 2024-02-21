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
#from hypy1d.flux_divergence import FluxDiv1DDGStrong as FluxDiv1DDG
#from hypy1d.flux_divergence import FluxDiv1DDGWeak as FluxDiv1DDG
from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGStrongSWE as FluxDiv1DDG
from hypy1d.limiter_none    import Limiter

def Initial(Grid, DGElmt, grav):

  Q = np.zeros((DGElmt.doflength,2))
  b = np.zeros(DGElmt.doflength)

  rm = 0.3
  hm = 0.2
  xm = 0.5

  xrel = DGElmt.dofcoordinates-xm
  mask = abs(xrel) < rm
  b[mask] = hm * np.exp(0.5/(xrel[mask]**2-rm**2)) / np.exp(0.5/(-rm**2))
  b = b - np.mean(b)

  #rm = 0.1
  #hm = 1.2
  #xm = 0.5

  #xrel = Grid.elementcenter-xm
  #mask = abs(xrel) < rm
  #b[DGElmt.elementdofs[mask]] = hm

  Q[:,0] = 1.0 - b
  mask = Q[:,0] < 0.0
  Q[mask,0] = 0.0
  Q[:,1] = 0.0 #1.0e-8*(np.random.rand(b.size)-0.5)

  mask = abs(DGElmt.dofcoordinates-0.2) < 0.03
  Q[mask,0] = Q[mask,0] + 0.05

  return Q, b


def PlotStep(Q, btopo, Q0, titletext):

  H = Q[:,0] + btopo
  dQ = Q[:,0] - Q0[:,0]

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(311)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

  ax1.set_ylabel('$h+b$')
  #ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  ax2.set_ylabel('$h$')
  ax2.axis(hrange)

  ax3 = fig.add_subplot(313)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax3.set_ylabel('$hu$')
  #ax3.axis(mrange)

  plt.draw()


N    = 1      # polynomial interpolation order
m    = 51     # number of grid nodes
xmin = 0.0    # position of left boundary
xmax = 1.0    # position of right boundary
Tmax = 0.2    # end time
CFL  = 0.90   # CFL number
dt   = 0.0005 # time step size
grav = 100.0  # gravitational constant
pltint = 10   # steps between plots
intpts = 2    # number of interpolation points within one element for visualisation

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

eq   = eq.EqSWERusanov(grav)
#eq    = eq.EqSWEHLLE(grav)
src   = EqSWESource(eq, dgel, dgqu)
FD    = FluxDiv1DDG(gr, eq, src, dgel, dgqu, [])
Lim   = Limiter(gr, eq, src, dgel)
RK    = RungeKuttaDG(FD, Lim, 2)

# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
fin = np.finfo(float)

# compute initial condition
Q, btopo = Initial(gr, dgel, grav)
src.update(btopo)

QQ = np.zeros((int(Tmax/dt)+1,dgel.doflength,2))
QQ[0] = Q

# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+1,gr.elength,2))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

hrange = [xmin, xmax, -0.1, 1.1]
mrange = [xmin, xmax, -0.001, 0.001]
urange = [xmin, xmax, -0.5, 0.5]

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
while t < Tmax-fin.resolution:
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

  print("step: {0:4d}, time: {1:6.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "mass error: {0:+6.4e}".format((Qm[s]-Qm[0]).mean(axis=0)[0]))

