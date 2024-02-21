"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid import Grid1D, generate_uniformgrid
from hypy1d.swe.riemann_solvers import EqSWEHLLE
from hypy1d.swe.equation    import EqSWEFVSource
from hypy1d.timestepping        import RungeKutta
from hypy1d.flux_divergence     import FluxDiv1DFVSrc
from hypy1d.fvrecovery.recovery import PiecewLinRecovery, PiecewLinRecoveryHydrostatic
import hypy1d.fvrecovery.limiters as lim
from hypy1d.fvboundary import boundary_flux
from hypy1d.fvboundary import boundary_value

def Initial(Grid, grav):
  """
  Definition of initial conditions: Lake at rest with non-trivial bottom topography
  """

  Q = np.zeros((Grid.elength,2))
  b = np.zeros(Grid.elength)

  #rm = 0.3
  #hm = 0.2
  rm = 0.1
  hm = 0.5
  xm = 0.5

  xrel = Grid.elementcenter-xm
  mask = abs(xrel) < rm
  #b[mask] = hm * np.exp(0.5/(xrel[mask]**2-rm**2)) / np.exp(0.5/(-rm**2))
  b[mask] = hm
  b = b - np.mean(b)

  Q[:,0] = 1.0 - b
  mask = Q[:,0] < 0.0
  Q[mask,0] = 0.0
  Q[:,1] = 0.0 #1.0e-8*(np.random.rand(b.size)-0.5)

  #mask = abs(Grid.elementcenter-0.2) < 0.03
  #Q[mask,0] = Q[mask,0] + 0.05

  return Q, b


def PlotStep(Q, btopo, Grid, titletext):

  # plot data
  fig.clf()

  ax1 = fig.add_subplot(311)
  ax1.plot(Grid.elementcenter, Q[:,0]+btopo)
  #ax1.axis(hrange1)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  ax2.plot(Grid.elementcenter, Q[:,0]+btopo)
  ax2.plot(Grid.elementcenter, btopo)
  ax2.axis(hrange2)
  ax2.set_title(titletext)

  ax3 = fig.add_subplot(313)
  ax3.plot(Grid.elementcenter, Q[:,1])
  ax3.axis(mrange)

  plt.draw()


m    = 101    # number of grid nodes
xmin = 0.0    # position of left boundary
xmax = 1.0    # position of right boundary
Tmax = 0.2    # end time
CFL  = 0.90   # CFL number
#dt   = 0.005  # time step size
dt   = 0.0005 # time step size
#grav = 1.0    # gravitational constant
grav = 100.0  # gravitational constant
pltint = 10   # steps between plots

ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
gr     = Grid1D(ndcoo, eltnd, ndels)

# compute initial condition
Q, btopo = Initial(gr, grav)

QQ = np.zeros((int(Tmax/dt)+1,gr.elength,2))
QQ[0] = Q

reclin = PiecewLinRecovery(gr, lim.LimMC, [])
rechyd = PiecewLinRecoveryHydrostatic(gr, lim.LimMC, [], btopo)

eq   = EqSWEHLLE(grav)
src  = EqSWEFVSource(eq, gr, reclin)
src.update(btopo)
FD   = FluxDiv1DFVSrc(gr, eq, rechyd, boundary_flux.BFluxInternal, src)
RK   = RungeKutta(FD, 2)

hrange1 = [xmin, xmax, 0.9, 1.1]
hrange2 = [xmin, xmax, -0.1, 1.1]
mrange = [xmin, xmax, -1.1, 1.2]

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

  print("step: {0:4d}, time: {1:7.4f}, dt = {2:6.4f}".format(s,t,dt))
