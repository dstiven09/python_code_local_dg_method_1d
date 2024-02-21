"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid import Grid1D, generate_uniformgrid
import hypy1d.swe.riemann_solvers as eq
from hypy1d.swe.equation    import EqSWEFVSource
from hypy1d.timestepping        import RungeKutta
from hypy1d.boundary_value      import BoundaryValueFVZeroExtrap
from hypy1d.flux_divergence     import FluxDiv1DFVSrc
from hypy1d.fvrecovery.recovery import PiecewLinRecovery, PiecewLinRecoveryHydrostatic
import hypy1d.fvrecovery.limiters as lim
from hypy1d.fvboundary import boundary_flux

def Initial(Grid, grav):
  """
  Definition of initial conditions: Tsunami Runup onto uniform sloping beach

  Note: Here we assume that the data is given at element boundaries!
  """

  Q = np.zeros((Grid.elength,2))
  b = np.zeros(Grid.elength)
  
  data  = np.loadtxt('data/Initial_runup.dat',skiprows=13)
  bathy = 0.1 * Grid.elementcenter
  btopo = 5000.0 - bathy

  for i in range(Grid.elength):
    if (Grid.elementcenter[i] > 0.0):
      indl = np.max(np.where(data[:,0]-Grid.elementcenter[i]<0))
      indr = np.min(np.where(data[:,0]-Grid.elementcenter[i]>0))
      Q[i,0] = (data[indl,1] + data[indr,1]) / 2.0 + bathy[i]

  return Q, btopo


def PlotStep(Q, btopo, Grid, titletext):

  # plot data
  fig.clf()

  ax1 = fig.add_subplot(311)
  ax1.plot(Grid.elementcenter, Q[:,0]+btopo, 'b-')
  ax1.plot(Grid.elementcenter, btopo, 'g')

  ax1.set_ylabel('$h+b$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  ax2.plot(Grid.elementcenter, Q[:,1], 'b-')

  ax2.set_ylabel('$hu$')
  ax2.axis(mrange)

  u = np.zeros(Grid.elength)
  mask = Q[:,0] > 1e-8
  u[mask] = Q[mask,1] / Q[mask,0]

  ax3 = fig.add_subplot(313)
  ax3.plot(Grid.elementcenter, u, 'b-')

  ax3.set_ylabel('$u$')
  ax3.axis(urange)

  plt.draw()


m    = 1011    # number of grid nodes
xmin = -500.0  # position of left boundary
xmax = 50000.0 # position of right boundary
Tmax =   220.0 # end time
CFL  = 0.90    # CFL number
dt   = 0.05    # time step size
grav = 9.81    # gravitational constant
pltint = 100   # steps between plots

ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr     = Grid1D(ndcoo, eltnd, ndels)

# compute initial condition
Q, btopo = Initial(gr, grav)
Q0 = np.copy(Q)

QQ = np.zeros((int(Tmax/dt)+1,gr.elength,2))
QQ[0] = Q

bv    = [BoundaryValueFVZeroExtrap(gr, gr.bnodes[0]),
         BoundaryValueFVZeroExtrap(gr, gr.bnodes[1])]
reclin = PiecewLinRecovery(gr, lim.LimMC, bv)
rechyd = PiecewLinRecoveryHydrostatic(gr, lim.LimMC, bv, btopo)

eq   = eq.EqSWERusanov(grav)
#eq   = eq.EqSWEHLLE(grav)
src  = EqSWEFVSource(eq, gr, reclin)
src.update(btopo)
FD   = FluxDiv1DFVSrc(gr, eq, rechyd, boundary_flux.BFluxInternal, src)
RK   = RungeKutta(FD, 2)

#hrange = [xmin, xmax, 4990.0, 5020.0]
#mrange = [xmin, xmax, -500.0,  400.0]
#urange = [xmin, xmax, -100.0,  100.0]
hrange = [-500.0, 1500.0, 4980.0, 5050.0]
mrange = [-500.0, 1500.0, -100.0, 100.0]
urange = [-500.0, 1500.0, -50.0,  50.0]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, btopo, gr, 'Initial conditions')
plt.show(block=False)

while not plt.waitforbuttonpress():
  pass

# loop over time
t = 0.0
s = 0
while t < Tmax:
  Qnew = RK.step(Q, t, dt)
  Q = Qnew
  t = t + dt
  s = s + 1
  QQ[s] = Q

  # plot data every pltint steps
  if (np.mod(s,pltint) == 0):
    PlotStep(Q, btopo, gr, 't =' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)

  print("step: {0:4d}, time: {1:6.2f}, dt = {2:4.2f}".format(s,t,dt))
