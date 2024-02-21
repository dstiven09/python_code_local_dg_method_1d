# -*- coding: utf-8 -*-
"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Susanne Beckers (2015)
"""

import matplotlib.pyplot as plt
import numpy as np
#import time
import math

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
from hypy1d.interpolation   import Vandermonde1D
from hypy1d.timestepping    import RungeKuttaDG


# import equation dependent expressions
from hypy1d.ade.equation_2ndDerivFlux    import EqADESource as EqSource
from hypy1d.ade.equation_2ndDerivFlux    import AnaSolutionStep as AnaSolution
from hypy1d.ade.equation_2ndDerivFlux    import Residual

# import flux computation
from hypy1d.flux_divergence_2ndDerivFlux_Diffusion import FluxDiv1DDGStrong \
                                                   as FluxDiv1DDG
#import numerical flux
import hypy1d.ade.riemann_solvers_2ndDerivFlux as eq

# import Limiter
from hypy1d.limiter_none    import Limiter

# import errors and goals
from hypy1d.ade.goals       import L2error as L2
from hypy1d.ade.goals       import H1error as H1
from hypy1d.ade.goals       import GoalDeriv


def Initial(Grid, DGElmt, grav):
  """
  Definition of initial condition: stepfunction
  """
  Q = np.zeros((DGElmt.doflength,1))
  b = np.zeros(DGElmt.doflength)

# Initialize step function
  for i in range(0, DGElmt.doflength):
     if (DGElmt.dofcoordinates[i] > 0.0):
        Q[i] = 0.0
     elif (DGElmt.dofcoordinates[i] < -1.0):
        Q[i] = 0.0
     else:
        Q[i] = 1.0

  return Q, b


def PlotStep(Q, QAna, Grid, titletext):

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(311)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt]]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt]]), 'bo')

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(312)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, QAna[dgel.elementdofs[ielt]]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], QAna[dgel.elementdofs[ielt]]), 'bo')

  ax2.set_ylabel('$h$')
  ax2.axis(hrange)
  ax2.set_title('Analytic solution')

  ax3 = fig.add_subplot(313)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    a = QAna[dgel.elementdofs[ielt]]
    b = np.dot(intpsi,Q[dgel.elementdofs[ielt]])
    ax3.plot(intx[0], b[0]-a[0], 'bo')

  ax3.set_ylabel('$\Delta h$')
  ax3.axis(hrangeDeltah)
  ax3.set_title('difference')

  plt.draw()


N      = 2      # polynomial interpolation order
m      = 3      # number of grid nodes [5,9,17,33,65,129, 257, 513, 1025]
xmin   = -1.0   # position of left boundary
xmax   = 1.0    # position of right boundary
Tmax   = 1.0    # end time
CFL    = 0.90   # CFL number
dt     = 0.001    # smallest time step size #0.0001
Tsteps = round(Tmax/dt) # total number of time steps
grav   = 100.0   # gravitational constant
pltint = 1000     # steps between plots
intpts = 3       # number of interpolation points within one element for visualisation
eps    = 0.1     # diffusion parameter


k = 0 #5
m = 3 #33
equ   = eq.EqADEUpWind(grav,eps)
print 'Zeitschritt dt=', dt
print 'Viskosität  eps=', eps


# loop over different refinements
while (m <10):
        # m=m+1
        k = k+1
        m = 2**k+1
        relt = DGReferenceElement1D(N)
        ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
        gr   = Grid1D(ndcoo, eltnd, ndels)
        dgel = DGElement1D(gr, relt)
        dgqu = DGQuadrature1D(relt, N)
        dgqu2 = DGQuadrature1D(relt, 40)

        src   = EqSource(equ, dgel, dgqu, eps)
        bv    = []
        res   = Residual(gr, dgel, dgqu, dgqu, eps)
        FD    = FluxDiv1DDG(gr, equ, src, dgel, dgqu, bv)
        Lim   = Limiter(gr, equ, src, dgel) # no limiter in use
        RK    = RungeKuttaDG(FD, Lim,2)

        # compute interpolation points and the mapping from the dofs for visualisation
        intxre  = np.linspace(-1.0, 1.0, intpts)
        intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
        ifstlst = [0, -1]
        fin = np.finfo(float)

        # compute initial condition
        Q, btopo = Initial(gr, dgel, grav)
        src.update(btopo)


        # plot initial condition
        hrange = [xmin, xmax, -1.2, 1.7]
        hrangeDeltah = [-1,1, -0.1, 0.1]
        fig = plt.figure(1)

        sol  = AnaSolution(0,eps)
        f    = sol.AnalyticDiscrete(dgel.dofcoordinates)

        PlotStep(Q[:,0], f, gr, 'Initial conditions')

        plt.show(block=False)
        plt.draw()

        #while not plt.waitforbuttonpress():
        # pass

        # loop over time
        t = 0.0
        s = 0

        # initialization for goal evaluation
          #L2_error = np.zeros(Tsteps)
          #H1_error = np.zeros(Tsteps)
        JPrime        = GoalDeriv(Q, gr, dgel, dgqu2, sol.AnalyticConti)
        Residu        = np.zeros(Tsteps+1)
        Residu[0]     = res.WeightedResidual(Q,t,Tmax)

        # loop over time
        while t < Tmax:#-fin.resolution:

            # computation of next step
            Q = RK.step(Q, t, dt)
            t = t + dt
            s = s + 1

            # evaluate goal functional
            sol = AnaSolution(t,eps)
            Residu[s] = res.WeightedResidual(Q,t,Tmax)

            # evaluate analytic solution
            f = sol.AnalyticDiscrete(dgel.dofcoordinates)

            # time dependant Dirichlet bounaries
            Q[0,0] = sol.AnalyticConti(xmin)
            Q[dgel.doflength-1,0] = sol.AnalyticConti(xmax)

            # plot data every pltint steps
            if(np.mod(s,pltint)==0):
                PlotStep(Q, f, gr, 't = ' + str(t) + ', step = ' + str(s))
                plt.pause(0.01)

        # end loop over time
        L2_error = L2(Q, gr, dgel, dgqu2, sol.AnalyticConti)
        H1_error = H1(Q, gr, dgel, dgqu, dgqu2, sol.AnalyticConti,\
                               sol.AnalyticDerivative)
        JPrime = JPrime - GoalDeriv(Q, gr, dgel, dgqu2, sol.AnalyticConti)
        WeightedBound = res.WeightedB(Q,Tmax,Tmax)-res.WeightedB(Q,0,Tmax)
        WeightedRes = math.fsum(Residu)*dt-WeightedBound-JPrime

        #print 'Change in goal functional :', goal[s-1]/goal[0], 'times inital value.'
        print 'Element width =', gr.elementwidth[1],'L2-error=', L2_error, \
                                   'H1-error=', H1_error, 'WeightedRes=', \
                                    WeightedRes

# end loop over refinement
