# -*- coding: utf-8 -*-
"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Susanne Beckers (2015)
"""

import matplotlib.pyplot as plt
import numpy as np
import math

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
from hypy1d.interpolation   import Vandermonde1D
from hypy1d.timestepping    import Euler

import hypy1d.diffusion_equation.riemann_solvers_2ndDerivFlux as eq
from hypy1d.diffusion_equation.equation_2ndDerivFlux    import EqSource
from hypy1d.flux_divergence_2ndDerivFlux_Diffusion import FluxDiv1DDGStrong\
                                                   as FluxDiv1DDG

#from hypy1d.limiter_wd      import Limiter
from hypy1d.limiter_none    import Limiter
from hypy1d.diffusion_equation.goals       import L1error as L1
from hypy1d.diffusion_equation.goals       import L2error as L2
from hypy1d.diffusion_equation.goals       import H1error as H1
from hypy1d.diffusion_equation.equation_2ndDerivFlux    import AnaSolution

def Initial(Grid, DGElmt, eps):
  """
  Definition of initial condition: stepfunction
  """
  Q = np.zeros((DGElmt.doflength,1))
  b = np.zeros(DGElmt.doflength)

# Initialize exponential function
  for i in range(0, DGElmt.doflength):
  # Reviere example
    #Q[i] = math.cos(8.0*math.pi*DGElmt.dofcoordinates[i])
  # x⁴ function
    #e = 100.0
    #Q[i] = e*1.0/eps*DGElmt.dofcoordinates[i]**4.0\
    #       -2.0*e/eps*DGElmt.dofcoordinates[i]**2.0+e
  # x² function
    Q[i] = -(DGElmt.dofcoordinates[i]+1)*(DGElmt.dofcoordinates[i]-1)

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
    ax3.plot(intx[0], b[1]-a[1], 'bo')

  ax3.set_ylabel('$\Delta h$')
  ax3.axis(hrangeDeltah)
  ax3.set_title('difference')

  plt.draw()


N      = 2      # polynomial interpolation order
m      = 5      # number of grid nodes [5,9,17,33,65,129, 257, 513, 1025]
xmin   = -1.0   # position of left boundary
xmax   = 1.0    # position of right boundary
Tmax   = 0.1    # end time
CFL    = 0.90   # CFL number
dt     = 0.001    # smallest time step size #0.0001
Tsteps = Tmax/dt # total number of time steps
grav   = 100.0   # gravitational constant
pltint = 10      # steps between plots
intpts = 3       # number of interpolation points within one element for visualisation
eps    = 1.0    # diffusion parameter


k = 0 #4
m = 3 # 17
equ   = eq.EqDEaverage(grav,eps)
print 'Zeitschritt dt=', dt
print 'Viskosität  eps=', eps


# loop over different refinements
while (m <6):
        k = k+1
        m = 2**k+1
        relt = DGReferenceElement1D(N)
        ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, 'Dirichlet',\
                               bcl=0, bcr=0)
        gr   = Grid1D(ndcoo, eltnd, ndels)
        dgel = DGElement1D(gr, relt)
        dgqu = DGQuadrature1D(relt, N)
        dgqu2 = DGQuadrature1D(relt, 10)

        src   = EqSource(equ, dgel, dgqu, eps)
        FD    = FluxDiv1DDG(gr, equ, src, dgel, dgqu)
        Lim   = Limiter(gr, equ, src, dgel) # no limiter in use
        #RK    = RungeKuttaDG(FD, Lim, 2)
        RK    = Euler(FD)

        # compute interpolation points and the mapping from the dofs for visualisation
        intxre  = np.linspace(-1.0, 1.0, intpts)
        intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
        ifstlst = [0, -1]
        fin = np.finfo(float)

        # compute initial condition
        Q, btopo = Initial(gr, dgel, eps)
        src.update(btopo)


        # plot initial condition
        hrange = [xmin, xmax, -0.5, 1.5]
        hrangeDeltah = [xmin,xmax, -0.1, 0.1]
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

        # loop over time
        while t < Tmax-fin.resolution:
            Q = RK.step(Q, t, dt)
            t = t + dt
            s = s + 1

            # evaluate goal functional
            sol      = AnaSolution(t,eps)


            # evaluate analytic solution
            f  = sol.AnalyticDiscrete(dgel.dofcoordinates)

            # time dependant Dirichlet bounaries
            #Q[0,0] = sol.AnalyticConti(xmin)
            #Q[dgel.doflength-1,0] = sol.AnalyticConti(xmax)

            # plot data every pltint steps
            if(np.mod(s,pltint)==0):
                PlotStep(Q, f, gr, 't = ' + str(t) + ', step = ' + str(s))
                plt.pause(0.01)

        # end loop over time
        L1_error = L1(Q, gr, dgel, dgqu2, sol.AnalyticConti)
        H1_error = H1(Q, gr, dgel, dgqu, dgqu2, sol.AnalyticConti,\
                               sol.AnalyticDerivative)
        #print 'Change in goal functional :', goal[s-1]/goal[0], 'times inital value.'
        print 'Element width =', gr.elementwidth[1],'L1-error=', L1_error, \
                                   'H1-error=', H1_error


# end loop over refinement
