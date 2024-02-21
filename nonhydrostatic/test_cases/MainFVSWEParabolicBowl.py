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

def AnalyticSolution(t, x, a, B, h0, btopo):

  om = np.sqrt(2.0*grav*h0) / a
  h  = h0 - btopo - B**2/(4.0*grav)*(np.cos(2.0*om*t) + 1.0) - \
       B*x/(2.0*a)*np.sqrt(8.0*h0/grav)*np.cos(om*t)
  h[h<0.0] = 0.0
  u  = np.zeros(h.shape)
  u[h>0.0] = B*a*om/np.sqrt(2.0*h0*grav)*np.sin(om*t)

  return h,u


def Initial(Grid, grav):
  """
  Definition of initial conditions: Parabolic bowl
  """

  Q = np.zeros((Grid.elength,2))
  b = np.zeros(Grid.elength)

  a  = 3000.0
  B  = 5.0
  h0 = 10.0

  btopo = h0*(Grid.elementcenter/a)**2

  h, u = AnalyticSolution(0.0, Grid.elementcenter, a, B, h0, btopo)
  Q[:,0] = h
  Q[:,1] = h*u

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


m    = 201     # number of grid nodes
xmin = -5000.0 # position of left boundary
xmax =  5000.0 # position of right boundary
Tmax =  1000.0 # end time
CFL  = 0.90    # CFL number
dt   = 1.0     # time step size
grav = 9.81    # gravitational constant
pltint = 20    # steps between plots

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
#reclin = PiecewLinRecovery(gr, lim.LimMinMod, bv)
#rechyd = PiecewLinRecoveryHydrostatic(gr, lim.LimMinMod, bv, btopo)

#eq   = eq.EqSWERusanov(grav)
eq   = eq.EqSWEHLLE(grav)
src  = EqSWEFVSource(eq, gr, reclin)
src.update(btopo)
FD   = FluxDiv1DFVSrc(gr, eq, rechyd, boundary_flux.BFluxInternal, src)
RK   = RungeKutta(FD, 2)

hrange = [xmin, xmax, 0.0, 20.0]
mrange = [xmin, xmax, -60.0, 60.0]
urange = [xmin, xmax, -20.0, 20.0]

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

  print("step: {0:4d}, time: {1:6.1f}, dt = {2:3.1f}, ".format(s,t,dt) +
        "mass error: {0:+6.4e}".format((Q-Q0).mean(axis=0)[0]))
