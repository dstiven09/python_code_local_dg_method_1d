"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)

This test case is an analytical solution to the non-hydrostatic extension for the shallow water eqations with quadratic vertical pressure profile as described in Jeschke (2017) and Sebra-Santos (1987).

Jeschke, A., Pedersen, G.K., Vater, S., Behrens, J.: Depth-averaged non-hydrostatic extension for shallow water equations with quadratic vertical pressure profile: Equivalence to boussinesq-type equations. International Journal for Numerical Methods in Fluids (2017). DOI:10.1002/fld.4361. URL http://dx.doi.org/10.1002/fld.4361.

Seabra-Santos, F.J., Renuoard, D.P., Temperville, A.M.: Numerical and experimental study of the transformation of a solitary wave over a shelf or isolated obstacle. Journal of Fluid Mechanics 176, 117-134 (1987)

This is the local version of the non-hydrostatic extension.

"""

import matplotlib.pyplot as plt
import numpy as np
import time
from decimal import *

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
import hypy1d.swe.riemann_solvers as eq
from hypy1d.interpolation   import Vandermonde1D
from hypy1d.swe.equation    import EqSWESource
from hypy1d.timestepping_r2   import RungeKuttaDG
from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap
#from hypy1d.flux_divergence import FluxDiv1DDGStrong as FluxDiv1DDG
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGStrongSWE as FluxDiv1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
from hypy1d.limiter_none    import Limiter

# non-hydrostatic extension
from hypy1d.elliptic.equation import EllipticEquation, AnalyticalSolution, Globalnh, Localnh
from hypy1d.factors_local     import FactorsElliptic, Solve
from hypy1d.boundary_elliptic import LeftDirichlet as Left
from hypy1d.boundary_elliptic import RightDirichlet as Right
#from hypy1d.boundary_elliptic_local import LeftPeriodic as Left    # for global computation: ilocal=1
#from hypy1d.boundary_elliptic_local import RightPeriodic as Right  # for global computation: ilocal=1


import time
start_time = time.time()


def multruns_input():

  value  = np.loadtxt('multruns_in')

  return value

def PlotStep(Q, Qm, fact, btopo, Grid, titletext):

  tol = 1.0e-8

  #H = btopo
  H = Q[:,0] + btopo
  #Hm = Qm[:,0] + btopo

  ihb = 0
  #ihb = 1

  if (A>0.):
    if (ihb==1):
      i1 = 511
      i2 = 512
      i3 = 513
    else:
      i1 = 0
      i2 = 411
      i3 = 412
  else:
    if (ihb==1):
      i1 = 311
      i2 = 312
      i3 = 313
    else:
      i1 = 0
      i2 = 211
      i3 = 212

  # plot data
  fig.clf()
  if (ihb==1):
    ax5 = fig.add_subplot(i1)
    for ielt in range(gr.elength):
      if (any(dgel.elementdofs[ielt,0]==fact.Local.dofloc)):
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax5.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'r-')
        ax5.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'ro')
      else:
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax5.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
        ax5.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

    #ax5.plot(Grid.elementcenter, Hm[:,0], 'ro')

    ax5.set_ylabel('$h+b$')
    ax5.axis(hbrange)
    ax5.set_title(titletext)

  ax1 = fig.add_subplot(i2)
  for ielt in range(gr.elength):
    if (any(dgel.elementdofs[ielt,0]==fact.Local.dofloc)):
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'r-')
      ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'ro')
    else:
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
      ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  #ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$h$')
  #ax1.axis(hrange)
  if (ihb==0):
    ax1.set_title(titletext)

  ax2 = fig.add_subplot(i3)
  for ielt in range(gr.elength):
    if (any(dgel.elementdofs[ielt,0]==fact.Local.dofloc)):
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'r-')
      ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'ro')
    else:
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
      ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

    #ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$hu$')
  #ax2.axis(hurange)

  if (A>0.):
    ax3 = fig.add_subplot(i3+1)
    for ielt in range(gr.elength):
      if (any(dgel.elementdofs[ielt,0]==fact.Local.dofloc)):
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],2]), 'r-')
        ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],2]), 'ro')
      else:
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],2]), 'b-')
        ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],2]), 'bo')

      #ax3.plot(Grid.elementcenter, Pm[:,0], 'ro')

    ax3.set_ylabel('$hw$')
    #ax3.axis(hwrange)

    ax4 = fig.add_subplot(i3+2)
    for ielt in range(gr.elength):
      if (any(dgel.elementdofs[ielt,0]==fact.Local.dofloc)):
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],3]), 'r-')
        ax4.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],3]), 'ro')
      else:
        intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
        ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],3]), 'b-')
        ax4.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],3]), 'bo')

      #ax4.plot(Grid.elementcenter, Pm[:,1], 'ro')

    ax4.set_ylabel('$pnh$')
    #ax4.axis(hpnhrange)

  plt.draw()


def Initial(Grid, DGElmt, anasol, iunknowns):
  """
  Definition of initial conditions: periodic standing wave
  """

  t = 0.
  Q = np.zeros((DGElmt.doflength,iunknowns))
  b = np.zeros(DGElmt.doflength)

  a = 2.
  x0 = (max(DGElmt.dofcoordinates)-min(DGElmt.dofcoordinates))/4.

  if(A>0.):
    Q[:,0], Q[:,1], Q[:,2], Q[:,3] = anasol.analytical_solit(0., a, x0)
  if(A==0.):
    Q[:,0], Q[:,1] = anasol.analytical_solit(0., a, x0)

  return Q, b, a, x0

# definition of set-up
ilocal    = 0      # flag for local run
nhcrit    = 1.001  # criterion definition for local run (see hypy1d/factors_local.py in routine local_arrays)
iconvtest    = 1   # flag for convergence test
itimesaveall = 0   # flag to same results at all time steps
iploterror   = 0   # flag to plot the error to analyical solution
timestamparr = np.array([5., 10., 15., 20., 30., 40., 50., 60., 70., 80., 90., 100.])  # times to take timestamps
#m = multruns_input()   # for convergence test
m    = 101      # number of grid nodes
d = 10.         # average water depth
swnl = 1        # flag for non-linear hydrostatic version
N    = 1        # polynomial interpolation order
xmin = 0.0      # position of left boundary
xmax = 800.0    # position of right boundary
Tmax = 20.0     # end time
CFL  = 0.90     # CFL number
dt   = 20./(m-1.)   # time step, s.t. CFL=const.
#grav = 100.0       # gravitational constant
grav = 9.80616      # gravitational constant
pltint = (m-1)/2    # steps between plots, at same times independent of dt
savint = (m-1)/2    # steps between saves
#pltint = 1         # steps between plots
#savint = 0          # steps between saves or no save of plots (savint=0)
intpts = 3          # number of interpolation points within one element for visualisation

# definition non-hydrostatic set-up
A    = 1.5     # flag for hydrostatic run (A=0.), or non-hydrostatic run with quadratic pressure profile (A=1.5) or linear pressure profile (A=2.)
B    = 0.      # variable fd
nhnl = 1       # flag for non-linear non-hydrostatic run
nht  = 1       # order of time discretization: first order (nht=1) or second order (nht=2)
iunknowns = 4  # number of iunknowns in Q
if(A==0.):
  A = 0.
  B = 0.
  nhnl = 0
  nht = 0
  iunknowns = 2

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

# compute initial condition
anasol = AnalyticalSolution(dgel, grav, d, A, B)
Q, btopo, a, x0 = Initial(gr, dgel, anasol, iunknowns)

# define folder to save figures and convergence results
stest = 'solitary_local'
if(A==0.):
  shelp = 'swe/'
if(A==1.5):
  shelp = 'nh2/'
if(A==2.):
  shelp = 'nh1/'
sfold = 'diag/' + stest + '/' + shelp
#for convergence test, just change 'sfold' to 'sfoldconv' in loop with timestamparr
#sfoldconv = 'diag/convergence/' + stest + '/' + shelp
sfoldconv = sfold
sfolderr  = 'diag/' + stest + '/' + shelp + 'error/'

# connecting classes for hydrostatic code
if(ilocal==1):
  localnh = Localnh(gr, dgel)
if(ilocal==0):
  localnh = Globalnh(gr, dgel)
eq   = eq.EqSWERusanov(localnh, d, grav, swnl, nht, A, B, 1.0e-8, iunknowns-2)
#eq   = eq.EqSWEHLLE(d, grav, swnl, nht, A, 1.0e-8, iunknowns-2)
src   = EqSWESource(eq, dgel, dgqu)
bv    = []
#bv    = [BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
         #BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
FD    = FluxDivergence1DDG(gr, eq, src, dgel, dgqu, bv)
Lim   = Limiter(gr, eq, src, dgel)
RK    = RungeKuttaDG(FD, Lim, 2)
src.update(btopo)

# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
fin = np.finfo(float)

# set-up of non-hydrostatic part including local run and boundary conditions
eqell = EllipticEquation(dgel, grav, d, btopo, A, B, nhnl, nht)
fact  = FactorsElliptic(gr, src, eqell, dgel, dgqu, localnh)
sol   = Solve()
if(ilocal==1):
  lnh, lelmt, dofsnh = sol.local_arrays(fact, FD, Q, Q, dgel, 0., nhcrit)  # for local run
  localnh.update_local(lnh, lelmt, dofsnh)
if(ilocal==0):
  localnh.update_local()
#diri  = np.array([1,1])  # put (non-zero) Dirichlet boundary data here
diri  = np.array([])     # in case of non-(non-zero)-Dirichlet boundary data: zero Dirichlet, Periodic, Reflection
bvell = [Left(gr, dgel, localnh), Right(gr, dgel, localnh), diri]

# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+3,gr.elength,iunknowns))
if (iploterror==1):
  Qmerr = np.zeros((int(Tmax/dt)+3,gr.elength,iunknowns))
  Qana = np.zeros(Q.shape)
  if(A>0.):
    Qana[:,0], Qana[:,1], Qana[:,2], Qana[:,3] = anasol.analytical_solit(0., a, x0)
  if(A==0.):
    Qana[:,0], Qana[:,1] = anasol.analytical_solit(0., a, x0)
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])
  if (iploterror==1):
    Qmerr[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], (Q-Qana)[dgel.elementdofs[ielt]])


shift = 0.01
hrange = [xmin, xmax, min(Q[:,0]), max(Q[:,0])]
hurange = [xmin, xmax, min(Q[:,1])-shift, max(Q[:,1])+shift]
if (A>0.):
  hwrange = [xmin, xmax, min(Q[:,2])-shift, max(Q[:,2])+shift]
  hpnhrange = [xmin, xmax, min(Q[:,3])-shift, max(Q[:,3])+shift]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, Qm[0], fact, btopo, gr, 'Initial conditions')

plt.show(block=False)
plt.draw()
if (savint>0):
  plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=0.png')
if (iploterror==1):
  PlotStep(Q-Qana, Qmerr[0], fact, btopo, gr, 'Initial conditions error')
  plt.savefig(sfolderr + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=0_error.png')

#while not plt.waitforbuttonpress():
  #pass

# loop over time
t = 0.0
s = 0
Q0 = Q

if (iconvtest==1):
  filestr = sfoldconv + stest + '_timestamp_N=' + str(N) + '_t=' + str(t) + '_m=' + str(m) + '.out'
  with open(filestr, "w") as f2:
    for i in range(len(Q[:,0])):
      if (A>0.):
        f2.write(str(Q[i,0]) + '     ' + str(Q[i,1]) + '     ' + str(Q[i,2]) + '     ' + str(Q[i,3]) + '     ' +'\n')
      if (A==0.):
        f2.write(str(Q[i,0]) + '     ' + str(Q[i,1]) +'\n')
  f2.close()


# get previous time step for second order computation
Qold = np.zeros(Q.shape)
if(A>0.):
  Qold[:,0], Qold[:,1], Qold[:,2], Qold[:,3] = anasol.analytical_solit(-dt, a, x0)
if(A==0.):
  Qold[:,0], Qold[:,1] = anasol.analytical_solit(-dt, a, x0)

while t < Tmax-fin.resolution:
  u = np.zeros(dgel.doflength)
  #mask = Q[:,0] > wettol
  #u[mask] = abs(Q[mask,1] / Q[mask,0])
  #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
  CFLg = np.max(u+np.sqrt(eq.g*Q[:,0]))*dt / np.min(gr.elementwidth)

  Qnew, Qhelp = RK.step(Q, Qold, t, dt)

  # non-hydrostatic extension: projection method
  if (A>0.):

    if (ilocal==1):
      lnh, lelmt, dofsnh = sol.local_arrays(fact, FD, Qnew, Q0, dgel, t, nhcrit)  # for local run
      localnh.update_local(lnh, lelmt, dofsnh)

    if (fact.Local.doflenloc>0):
      hu, dipnh  = sol.Solve_sys(fact, Qnew, dt, t, bvell, Qhelp)     # solve lin. equ. sys., hu is already updated
      Qnew = sol.corr_tracer(fact, FD, Qnew, hu, dipnh, t, dt, Qhelp) # correction step
      Qnew = Lim(Qnew)  # application of limiter because correction step for water height

    #for plotting error between analytical and numerical solution
    if (iploterror==1):
      if(A>0.):
        Qana[:,0], Qana[:,1], Qana[:,2], Qana[:,3] = anasol.analytical_solit(t+dt, a, x0)
      if(A==0.):
        Qana[:,0], Qana[:,1] = anasol.analytical_solit(t+dt, a, x0)

  Qold = Q
  Q = Qnew

  hrange = [xmin, xmax, min(Q[:,0])-shift, max(Q[:,0])+shift]
  hurange = [xmin, xmax, min(Q[:,1])-shift, max(Q[:,1])+shift]
  if (A>0.):
    hwrange = [xmin, xmax, min(Q[:,2])-shift, max(Q[:,2])+shift]
    hpnhrange = [xmin, xmax, min(Q[:,3])-shift, max(Q[:,3])+shift]

  t = t + dt
  s = s + 1

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])
    if (iploterror==1):
      Qmerr[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], (Q-Qana)[dgel.elementdofs[ielt]])

  # plot data every pltint steps
  if ((np.mod(s,pltint)==0) ):# or (t<1.)):
    if (iploterror==1):
      PlotStep(Q-Qana, Qmerr[s], fact, btopo, gr, 't = ' + str(t) + ', step = ' + str(s) + ', error plot')
      plt.savefig(sfolderr + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=' + str(s) + '_error.png')
    PlotStep(Q, Qm[s], fact, btopo, gr, 't = ' + str(t) + ', step = ' + str(s) + ', CFL = ' + str(round(CFLg,3)))
    plt.pause(0.02)

  # save data every savint steps
  if ((np.mod(s,savint)==0) and (savint>0)):
    plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=' + str(s) + '.png')
  if (iconvtest==1):
    if (any(np.abs(timestamparr-t)<dt) or (itimesaveall==1)):
      filestr = sfoldconv + stest + '_timestamp_N=' + str(N) + '_t=' + str(t) + '_m=' + str(m) + '.out'
      with open(filestr, "w") as f2:
	for i in range(len(Q[:,0])):
	  if (A>0.):
	    f2.write(str(Q[i,0]) + '     ' + str(Q[i,1]) + '     ' + str(Q[i,2]) + '     ' + str(Q[i,3])  + '\n')
	  if (A==0.):
	    f2.write(str(Q[i,0]) + '     ' + str(Q[i,1]) +'\n')
      f2.close()

  print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "CFL = {0:4.2f}, ".format(CFLg) +
        "mass error: {0:+6.4e}, ".format((Qm[s]-Qm[0]).mean(axis=0)[0]) +
        "moment. error: {0:+6.4e}".format((Qm[s]-Qm[0]).mean(axis=0)[1]))

  if (np.isnan(CFLg)):
    quit()

print("--- %s seconds ---" % (time.time() - start_time))
