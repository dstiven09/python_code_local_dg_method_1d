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

def AnalyticSolution(t, x, a, B, h0, btopo):

  om = np.sqrt(2.0*grav*h0) / a
  h  = h0 - btopo - B**2/(4.0*grav)*(np.cos(2.0*om*t) + 1.0) - \
       B*x/(2.0*a)*np.sqrt(8.0*h0/grav)*np.cos(om*t)
  h[h<0.0] = 0.0
  u  = np.zeros(h.shape)
  u[h>0.0] = B*a*om/np.sqrt(2.0*h0*grav)*np.sin(om*t)

  return h,u


def Initial(Grid, DGElmt, grav):

  Q = np.zeros((DGElmt.doflength,2))

  a  = 3000.0
  B  = 5.0
  h0 = 10.0

  btopo = h0*(DGElmt.dofcoordinates/a)**2

  h, u = AnalyticSolution(0.0, DGElmt.dofcoordinates, a, B, h0, btopo)
  Q[:,0] = h
  Q[:,1] = h*u

  return Q, btopo


def PlotStep(Q, btopo, titletext):

  H = Q[:,0] + btopo

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(221)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'r-')
      ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'ro')
    else:
      ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
      ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

  ax1.set_ylabel('$h+b$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax1a = fig.add_subplot(222)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax1a.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'r-')
      ax1a.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'ro')
    else:
      ax1a.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
      ax1a.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  ax1a.set_ylabel('$h$')
  ax1a.axis([xmin, xmax, 0.0, 1e-4])

  ax2 = fig.add_subplot(223)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    Qelt  = Q[dgel.elementdofs[ielt]]
    belt  = btopo[dgel.elementdofs[ielt]]
    xhmax = np.argmax(Qelt[:,0])
    if (np.min(Qelt[:,0]) < wettol and Qelt[xhmax,0]+belt[xhmax]-np.max(belt) < wettol):
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'r-')
      ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'ro')
    else:
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
      ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax2.set_ylabel('$hu$')
  ax2.axis(mrange)

  ax3 = fig.add_subplot(224)
  for ielt in range(gr.elength):
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
      ax3.plot(intx[ifstlst], uE[ifstlst], 'ro')
    else:
      ax3.plot(intx, uE, 'b-')
      ax3.plot(intx[ifstlst], uE[ifstlst], 'bo')

  ax3.set_ylabel('$u$')
  ax3.axis(urange)

  plt.draw()


N    = 1       # polynomial interpolation order
m    = 201     # number of grid nodes
xmin = -5000.0 # position of left boundary
xmax =  5000.0 # position of right boundary
Tmax =  1000.0 # end time
CFL  = 0.16    # CFL number
dt   = 1.0     # time step size
grav = 9.81    # gravitational constant
wettol = 1e-08 # wet tolerance
pltint = 100   # steps between plots
intpts = 2     # number of interpolation points within one element for visualisation

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

hrange = [xmin, xmax, 0.0, 20.0]
mrange = [xmin, xmax, -60.0, 60.0]
urange = [xmin, xmax, -20.0, 20.0]

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
while t < Tmax-fin.resolution:
  u = np.zeros(dgel.doflength)
  mask = Q[:,0] > wettol
  u[mask] = abs(Q[mask,1] / Q[mask,0])
  #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
  CFLg = np.max(u+np.sqrt(eq.g*Q[:,0]))*dt / np.min(gr.elementwidth)
  CFLu = np.max(u)*dt / np.min(gr.elementwidth)

  Qnew = RK.step(Q, t, dt)
  Q = Qnew
  if(np.isnan(np.min(Qnew))):
    break
  t = t + dt
  s = s + 1
  QQ[s] = Q

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

  # plot data every pltint steps
  if (np.mod(s,pltint) == 0):
    PlotStep(Q, btopo, 't = ' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)
    #while not plt.waitforbuttonpress():
      #pass

  print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "CFLg = {0:4.2f}, CFLu = {1:4.2f}, ".format(CFLg,CFLu) +
        "mass error: {0:+6.4e}".format((Qm[s]-Qm[0]).mean(axis=0)[0]))
