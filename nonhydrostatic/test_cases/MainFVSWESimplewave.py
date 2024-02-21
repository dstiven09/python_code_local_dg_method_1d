"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid import Grid1D, generate_uniformgrid
from hypy1d.swe.riemann_solvers import EqSWEHLLE
from hypy1d.timestepping        import RungeKutta
from hypy1d.boundary_value      import BoundaryValueFVZeroExtrap
from hypy1d.flux_divergence     import FluxDiv1DFV
from hypy1d.fvrecovery.recovery import PiecewConstRecovery, PiecewLinRecovery
import hypy1d.fvrecovery.limiters as lim
from hypy1d.fvboundary import boundary_flux

def Initial(Grid, grav):
  """
  Definition of initial conditions: left going simple wave
  """

  Q = np.zeros((Grid.elength,2))
  b = np.zeros(Grid.elength)

#  for i in range(gr.elength):
#    if (gr.elementcenter[i] >=0.3 and gr.elementcenter[i] <=0.7):
#      Q[i,0] = 1.01
#    else:
#      Q[i,0] = 1.0
#    Q[i,1] = Q[i,0] * 0.0

  c0 = np.sqrt(grav)
  c1 = 0.5
  c  = c0 + c1*np.sin(2.0*np.pi*Grid.elementcenter)
  h  = c**2 / grav
  u  = 2.0 * (c-c0)

  Q[:,0] = h
  Q[:,1] = h*u

  return Q, b


def PlotStep(Q, Grid, titletext):

  # plot data
  fig.clf()

  ax1 = fig.add_subplot(311)
  ax1.plot(Grid.elementcenter, Q[:,0])

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  ax2.plot(Grid.elementcenter, Q[:,1])

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


m    = 101    # number of grid nodes
xmin = 0.0    # position of left boundary
xmax = 1.0    # position of right boundary
Tmax = 0.2    # end time
CFL  = 0.90   # CFL number
#dt   = 0.005  # time step size
dt   = 0.0005 # time step size
#grav = 1.0    # gravitational constant
grav = 100.0  # gravitational constant
pltint = 5    # steps between plots

ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
#ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
# compute initial condition
Q, btopo = Initial(gr, grav)

QQ = np.zeros((int(Tmax/dt)+1,gr.elength,2))
QQ[0] = Q

#bv    = [BoundaryValueFVZeroExtrap(gr, gr.bnodes[0]),
         #BoundaryValueFVZeroExtrap(gr, gr.bnodes[1])]
rec  = PiecewLinRecovery(gr, lim.LimMC, [])
#rec  = PiecewLinRecovery(gr, lim.LimMC, bv)
#rec  = PiecewConstRecovery(gr)

eq   = EqSWEHLLE(grav)
FD   = FluxDiv1DFV(gr, eq, rec, boundary_flux.BFluxInternal)
RK   = RungeKutta(FD, 2)

#hrange = [xmin, xmax, 0.98, 1.02]
#mrange = [xmin, xmax, -0.1, 0.1]
hrange = [xmin, xmax, 0.88, 1.12]
mrange = [xmin, xmax, -1.1, 1.2]
urange = [xmin, xmax, -1.1, 1.2]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, gr, 'Initial conditions')
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
    PlotStep(Q, gr, 't =' + str(t) + ', step = ' + str(s))
    plt.pause(0.02)

  print("step: {0:4d}, time: {1:7.4f}, dt = {2:5.4f}".format(s,t,dt))
