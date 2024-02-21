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
from hypy1d.boundary_value  import BoundaryValueDG, BoundaryValueDGZeroExtrap
from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
from hypy1d.limiter_bjsvhy1  import Limiter


class BoundaryValueDGInflowOutflow(BoundaryValueDG):
  """
  boundary value for DG discretization (preliminary inflow/outflow)
  """

  def __init__(self, Grid, bnode, DGElmt):

    super(BoundaryValueDGInflowOutflow, self).__init__(Grid, bnode, DGElmt)
    self.data = np.loadtxt('WaveFlumeDraehne/Zeitserien_T30_H2a.txt',skiprows=1)

  def __call__(self, Q, t):
    """
    evaluate boundary value from state Q at time t
    """

    Qb    = np.zeros(3)
    Qb[0] = 0.3 + np.interp(t+1.0, self.data[:,0], self.data[:,3]-self.data[100,3])/100.0
    Qb[1] = Qb[0]*np.interp(t+1.0, self.data[:,0], self.data[:,6]-self.data[100,6])/100.0
    #Qb[1] = 2.0*Qb[0]*(np.sqrt(grav*Qb[0]) - np.sqrt(grav*0.3))
    return Qb


def Initial(Grid, DGElmt, grav):

  Q = np.zeros((DGElmt.doflength,2))
  b = np.zeros(DGElmt.doflength)

  b      = np.maximum(0.0, 0.025*(DGElmt.dofcoordinates - 15.72))
  Q[:,0] = np.maximum(0.0, 0.3-b)
  return Q, b


def PlotStep(Q, btopo, titletext):

  H = Q[:,0] + btopo

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(311)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')
    ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')

  ax1.axis(hrange)
  ax1.set_ylabel('$h+b$')
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')

  ax2.set_ylabel('$hu$')
  ax2.axis(mrange)

  ax3 = fig.add_subplot(313)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    hE = np.dot(intpsi, Q[dgel.elementdofs[ielt],0])
    mE = np.dot(intpsi, Q[dgel.elementdofs[ielt],1])
    uE       = np.zeros(hE.shape)
    mask     = (hE > wettol)
    uE[mask] = mE[mask]/hE[mask]
    ax3.plot(intx, uE, 'b-')

  ax3.set_ylabel('$u$')
  ax3.axis(urange)

  plt.draw()


def PlotStepPart(Q, btopo, titletext):

  eltl = 0
  eltr = 1000
  H = Q[:,0] + btopo

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(311)
  for ielt in range(eltl, eltr):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'r-')
      #ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'ro')
    else:
      ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
      #ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

  ax1.set_ylabel('$h+b$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  for ielt in range(eltl, eltr):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'r-')
      #ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'ro')
    else:
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
      #ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax2.set_ylabel('$hu$')
  ax2.axis(mrange)

  ax3 = fig.add_subplot(313)
  for ielt in range(eltl, eltr):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    hE = np.dot(intpsi, Q[dgel.elementdofs[ielt],0])
    mE = np.dot(intpsi, Q[dgel.elementdofs[ielt],1])
    uE       = np.zeros(hE.shape)
    mask     = (hE > wettol)
    uE[mask] = mE[mask]/hE[mask]
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax3.plot(intx, uE, 'r-')
      #ax3.plot(intx[ifstlst], uE[ifstlst], 'ro')
    else:
      ax3.plot(intx, uE, 'b-')
      #ax3.plot(intx[ifstlst], uE[ifstlst], 'bo')

  ax3.set_ylabel('$u$')
  ax3.axis(urange)

  plt.draw()


N    = 1       # polynomial interpolation order
m    = 129     # number of grid nodes
xmin = 0.0     # position of left boundary
xmax = 32.0    # position of right boundary
Tmax = 80      # end time
CFL  = 0.90    # CFL number
dt   = 0.025   # time step size
grav = 9.81    # gravitational constant
wettol = 1e-8  # wet tolerance
pltint = 80    # steps between plots
intpts = 2     # number of interpolation points within one element for visualisation

hrange = [xmin, xmax, 0.27, 0.35]
mrange = [xmin, xmax, -0.02, 0.02]
urange = [xmin, xmax, -0.2, 0.2]

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

eq    = eq.EqSWERusanov(grav, wettol)
src   = EqSWESource(eq, dgel, dgqu)
bv    = [BoundaryValueDGInflowOutflow(gr, gr.bnodes[0], dgel),
         BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
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

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, btopo, 'Initial conditions')

plt.show(block=False)
plt.draw()

while not plt.waitforbuttonpress():
  pass

# loop over time
t = 0.0
s = 0
while s < smax:
  u = np.zeros(dgel.doflength)
  mask = Q[:,0] > wettol
  u[mask] = abs(Q[mask,1] / Q[mask,0])
  #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
  CFLg = np.max(u+np.sqrt(eq.g*Q[:,0]))*dt / np.min(gr.elementwidth)
  CFLu = np.max(u)*dt / np.min(gr.elementwidth)

  Qnew = RK.step(Q, t, dt)
  Q = Qnew
  t = t + dt
  s = s + 1
  QQ[s] = Q

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

  ## plot data every pltint steps
  if (np.mod(s,pltint)==0):
    PlotStep(Q, btopo, 't = ' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)

  print("step: {0:4d}, time: {1:7.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "CFLg = {0:4.2f}, CFLu = {1:4.2f}".format(CFLg,CFLu))

#np.save('WaveFlumeDraehne/QQ_DG_FDSWEWeak_LimBJSVhy1_tol1e-08_m203_dt0_05.npy',QQ)
