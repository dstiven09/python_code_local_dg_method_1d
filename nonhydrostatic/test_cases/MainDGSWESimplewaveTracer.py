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
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGStrongSWE as FluxDiv1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
from hypy1d.limiter_none    import Limiter

def Initial(Grid, DGElmt, grav):
  """
  Definition of initial conditions: left going simple wave
  """

  Q = np.zeros((DGElmt.doflength,3))
  b = np.zeros(DGElmt.doflength)

  c0 = np.sqrt(grav)
  c1 = 0.5
  c  = c0 + c1*np.sin(2.0*np.pi*DGElmt.dofcoordinates)
  h  = c**2 / grav
  u  = 2.0 * (c-c0)

  #Q[:,0] = h
  #Q[:,1] = h*u
  Q[:,0] = 1.0
  Q[:,1] = 10.0
  Q[:,2] = h*c

  return Q, b


def PlotStep(Q, Qm, Grid, titletext):

  tol = 1.0e-8

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(411)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(412)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$hu$')
  ax2.axis(mrange)

  ax3 = fig.add_subplot(413)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    hE = np.dot(intpsi, Q[dgel.elementdofs[ielt],0])
    mE = np.dot(intpsi, Q[dgel.elementdofs[ielt],1])
    uE       = np.zeros(hE.shape)
    mask     = (hE > tol)
    uE[mask] = mE[mask]/hE[mask]
    ax3.plot(intx, uE, 'b')
    ax3.plot(intx[ifstlst], uE[ifstlst], 'bo')

  ax3.set_ylabel('$u$')
  ax3.axis(urange)

  ax4 = fig.add_subplot(414)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    hE  = np.dot(intpsi, Q[dgel.elementdofs[ielt],0])
    hpE = np.dot(intpsi, Q[dgel.elementdofs[ielt],2])
    pE       = np.zeros(hE.shape)
    mask     = (hE > tol)
    pE[mask] = hpE[mask]/hE[mask]
    ax4.plot(intx, pE, 'b')
    ax4.plot(intx[ifstlst], pE[ifstlst], 'bo')

  ax4.set_ylabel('$\phi$')
  #ax4.axis(mrange)

  plt.draw()


N    = 3      # polynomial interpolation order
m    = 5      # number of grid nodes
xmin = 0.0    # position of left boundary
xmax = 1.0    # position of right boundary
Tmax = 0.1    # end time
CFL  = 0.90   # CFL number
dt   = 0.001  # time step size
grav = 100.0  # gravitational constant
pltint = 1    # steps between plots
intpts = 11   # number of interpolation points within one element for visualisation

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

#eq   = eq.EqSWERusanov(grav, 1e-8, 1)
eq    = eq.EqSWEHLLE(1., grav, 1, 1e-8, 1)
src   = EqSWESource(eq, dgel, dgqu)
bv    = []
#bv    = [BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
         #BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
FD    = FluxDivergence1DDG(gr, eq, src, dgel, dgqu, bv)
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

# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+1,gr.elength,3))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

hrange = [xmin, xmax, 0.88, 1.12]
mrange = [xmin, xmax, -1.1, 1.2]
urange = [xmin, xmax, -1.5, 1.5]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, Qm[0], gr, 'Initial conditions')

plt.show(block=False)
plt.draw()

while not plt.waitforbuttonpress():
  pass

# loop over time
t = 0.0
s = 0
while t < Tmax-fin.resolution:
  u = np.zeros(dgel.doflength)
  #mask = Q[:,0] > wettol
  #u[mask] = abs(Q[mask,1] / Q[mask,0])
  #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
  CFLg = np.max(u+np.sqrt(eq.g*Q[:,0]))*dt / np.min(gr.elementwidth)

  Qnew = RK.step(Q, t, dt)
  Q = Qnew
  t = t + dt
  s = s + 1

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

  # plot data every pltint steps
  if (np.mod(s,pltint)==0):
    PlotStep(Q, Qm[s], gr, 't = ' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)

  print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "CFL = {0:4.2f}, ".format(CFLg) +
        "mass error: {0:+6.4e}, ".format((Qm[s]-Qm[0]).mean(axis=0)[0]) +
        "moment. error: {0:+6.4e}".format((Qm[s]-Qm[0]).mean(axis=0)[1]))
