"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)

This test case follows the description of problem 2 and 5 given in
https://github.com/rjleveque/nthmp-benchmark-problems

"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
import hypy1d.swe.riemann_solvers as eq
from hypy1d.interpolation   import Vandermonde1D
from hypy1d.swe.equation    import EqSWESource
from hypy1d.timestepping_r2 import RungeKuttaDG
from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap, BoundaryValueDGWall
#from hypy1d.flux_divergence import FluxDiv1DDGStrong as FluxDiv1DDG
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGStrongSWE as FluxDiv1DDG
#from hypy1d.flux_divergenceSWE import FluxDiv1DDGWeakSWE as FluxDiv1DDG
from hypy1d.limiter_none    import Limiter

# non-hydrostatic extension
from hypy1d.elliptic.equation import EllipticEquation, AnalyticalSolution, Localnh, Globalnh
from hypy1d.factors           import FactorsElliptic, Solve
from hypy1d.boundary_elliptic import LeftDirichlet as Left
from hypy1d.boundary_elliptic import RightReflection as Right

# interpolation from first data points onto second data points
def func_interpolate(x1,x2,y1):
  y2 = np.zeros(len(x2))
  for i in range(len(x2)):
    m = 0.
    for j in range(len(x1)-1):
      if ((x1[j]<=x2[i]) and (x2[i]<=x1[j+1])):
        m = (y1[j+1]-y1[j])/(x1[j+1]-x1[j])
        break
    y2[i] = m*(x2[i]-x1[j])+y1[j]
  return y2


def multruns_input():

  value  = np.loadtxt('multruns_in')

  return value

def PlotStep(Q, Qm, btopo, Grid, titletext):

  tol = 1.0e-8

  #H = btopo
  H = Q[:,0] + btopo
  #Hm = Qm[:,0] + btopo

  #ihb = 0
  ihb = 1

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
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax5.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
      ax5.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

    #ax5.plot(Grid.elementcenter, Hm[:,0], 'ro')

    ax5.set_ylabel('$h+b$')
    ax5.axis(hbrange)
    ax5.set_title(titletext)

  ax1 = fig.add_subplot(i2)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  #ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  if (ihb==0):
    ax1.set_title(titletext)

  ax2 = fig.add_subplot(i3)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

    #ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$hu$')
  ax2.axis(hurange)

  if (A>0.):
    ax3 = fig.add_subplot(i3+1)
    for ielt in range(gr.elength):
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],2]), 'b-')
      ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],2]), 'bo')

      #ax3.plot(Grid.elementcenter, Qm[:,2], 'ro')

    ax3.set_ylabel('$hw$')
    ax3.axis(hwrange)

    ax4 = fig.add_subplot(i3+2)
    for ielt in range(gr.elength):
      intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
      ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],3]), 'b-')
      ax4.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],3]), 'bo')

      #ax4.plot(Grid.elementcenter, Qm[:,3], 'ro')

    ax4.set_ylabel('$pnh$')
    ax4.axis(hpnhrange)

  plt.draw()

def Initial(DGElmt, anasol, iunknowns, swnl, a, rlen, xshift):
  """
  Definition of initial conditions: periodic standing wave
  """

  Q = np.zeros((DGElmt.doflength,iunknowns))
  b = np.zeros(DGElmt.doflength)

  #well-balanced test
  #a = 0.

  if(A>0.):
    Q[:,0], Q[:,1], Q[:,2], Q[:,3], b = anasol.initial_compositebeach(0., swnl, a, rlen, xshift)
    #Q[:,0], Q[:,1], Q[:,2], Q[:,3] = anasol.analytical_solit(-dt, a, xmax - rlen - 4.36 - 2.93 - 0.9 - xshift)
  if(A==0.):
    Q[:,0], Q[:,1], b = anasol.initial_compositebeach(0., swnl, a, rlen, xshift)

  return Q, b


# definition of set-up
iconvtest = 0
timestamparr = np.array([0.05, 1., 5., 10., 15., 20.])
swnl      = 0        # flag for non-linear hydrostatic version: 1 (fully non-linear), 2 (without advection term), 0 (fully linear, taking btopo into account (choosing h=d-b as linearized depth))

d      = 0.218    # constant depth offshore
a      = 0.039    # amplitude (Case A = 0.05; Case B = 0.30 ; Case C = 0.70), but scaled with ...
rlen   = 2.4      # additional length of channel (Case A = 2.40; Case B = 0.98 ; Case C = 0.64)
xshift = 0.       # s.t. initial condition is not affected by bathmetry

N    = 1      # polynomial interpolation order
#m    = multruns_input()
m    = 201      # number of grid nodes, same refinement as in TAM (reflev=3 dort)
xmin = 0.0    # position of left boundary
xmax = 24.0 + 6.   # position of right boundary
Tmax = 20.0 + xshift/0.626  # end time
CFL  = 0.90   # CFL number
#dt   = 0.02  # time step size
dt = 4./(m-1.)
#grav = 100.0  # gravitational constant
grav = 9.80616  # gravitational constant
pltint = (m-1)/8   # steps between plots
#savint = (m-1)/8   # steps between saves
#pltint = 5         # steps between plots
savint = 0         # steps between saves or no save of plots (savint=0)
intpts = 1         # number of interpolation points within one element for visualisation

# definition non-hydrostatic set-up
A    = 1.5 # flag for hydrostatic run (A=0.), or non-hydrostatic run with quadratic pressure profile (A=1.5) or linear pressure profile (A=2.)
B    = 0.  # variable fd
nhnl = 0   # flag for non-linear non-hydrostatic run
nht  = 2   # order of time discretization: first order (nht=1) or second order (nht=2)
iunknowns = 4  # number of iunknowns in Q
if(A==0.):
  A = 0.
  B = 0
  nhnl = 0
  nht = 0
  iunknowns = 2

diagpoints = xmax-np.array([0.0, 0.43, 0.9, 2.37, 3.83, 6.01, 8.19, 10.59])

# establish grid and DG elements
relt = DGReferenceElement1D(N)
#ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

# compute initial condition
anasol = AnalyticalSolution(dgel, grav, d, A, B)
Q, btopo = Initial(dgel, anasol, iunknowns, swnl, a, rlen, xshift)

# define folder to save figures and convergence results
stest = 'composite'
if(A==0.):
  shelp = 'swe/'
if(A==1.5):
  shelp = 'nh2/'
if(A==2.):
  shelp = 'nh1/'
sfold = 'diag/' + stest + '/' + shelp
#sfoldconv = 'diag/convergence/' + stest + '/' + shelp
sfoldconv = sfold

# connecting classes for hydrostatic code
localnh = Globalnh(gr, dgel)
eq   = eq.EqSWERusanov(localnh, d, grav, swnl, nht, A, B, 1.0e-8, iunknowns-2)
#eq   = eq.EqSWEHLLE(d, grav, swnl, nht, A, 1.0e-8, iunknowns-2)
src   = EqSWESource(eq, dgel, dgqu)
#bv    = []
bv    = [BoundaryValueDGWall(gr, gr.bnodes[0], dgel),
         BoundaryValueDGWall(gr, gr.bnodes[1], dgel)]
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
fact = FactorsElliptic(gr, src, eqell, dgel, dgqu)
sol  = Solve()
localnh.update_local()
# boundary condition on left and right boundary for the unknown fo the elliptic problem
#diri  = np.array([1,1])  # put (non-zero) Dirichlet boundary data here
diri  = np.array([])     # in case of non-(non-zero)-Dirichlet boundary data: zero Dirichlet, Periodic, Reflection
bvell = [Left(gr, dgel, localnh), Right(gr, dgel, localnh), diri]


# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+3,gr.elength,iunknowns))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])


h0 = d
hbrange = [xmin, xmax, h0-a/2., h0+a]
hrange = [xmin, xmax, 0.0, 0.26]
hurange = [xmin, xmax, -0.03, 0.07]
hwrange = [xmin, xmax, -0.006, 0.006]
hpnhrange = [xmin, xmax, -0.02, 0.01]

# plot initial condition
fig = plt.figure(1)
PlotStep(Q, Qm[0], btopo, gr, 'Initial conditions')

plt.show(block=False)
plt.draw()
if (savint>0):
  plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=0.png')

# save initial timeseries at diagpoints
if (savint>0):
  interpol = np.zeros((iunknowns,len(diagpoints)))
  for j in range(Q.shape[1]):
    add = np.zeros(Q.shape[0])
    #if (j==0):
      #add = btopo[:]-h0
    interpol[j] = func_interpolate(dgel.dofcoordinates,diagpoints,Q[:,j] + add)
  for j in range(len(diagpoints)):
    filestr = sfoldconv + stest + '_timeseries_x=' + str(diagpoints[j]) + '.out'
    with open(filestr, "w") as f2:
      swrite = ''
      for k in range(Q.shape[1]-1):
	swrite = swrite + str(interpol[k,j]) + '     '
      swrite = swrite + str(interpol[k,j]) + '\n'
      f2.write(swrite)
      f2.close()


#while not plt.waitforbuttonpress():
  #pass

# loop over time
t = 0.0
s = 0

# get previous time step for second order computation
Qold = np.zeros(Q.shape)
#Qold[:,0], Qold[:,1], Qold[:,2], Qold[:,3], btopo = anasol.initial_compositebeach(-dt, a, rlen, xshift)


while t < Tmax-fin.resolution:
  u = np.zeros(dgel.doflength)
  #mask = Q[:,0] > wettol
  #u[mask] = abs(Q[mask,1] / Q[mask,0])
  #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
  CFLg = np.max(u+np.sqrt(eq.g*Q[:,0]))*dt / np.min(gr.elementwidth)

  Qnew, Qhelp = RK.step(Q, Qold, t, dt)

  # non-hydrostatic extension: projection method
  if (A>0.):
    hu, dipnh  = sol.Solve_sys(fact, Qnew, dt, t, bvell, Qhelp)     # solve lin. equ. sys., hu is already updated
    Qnew = sol.corr_tracer(fact, FD, Qnew, hu, dipnh, t, dt, Qhelp) # correction step
    Qnew = Lim(Qnew)  # application of limiter because correction step for water height

  Qold = Q
  Q = Qnew

  t = t + dt
  s = s + 1

  # compute cell mean values
  for ielt in range(gr.elength):
    Qm[s,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

  # plot data every pltint steps
  if (np.mod(s,pltint)==0):
    PlotStep(Q, Qm[s], btopo, gr, 't = ' + str(t) + ', step = ' + str(s) + ', CFL = ' + str(round(CFLg,3)))
    #plt.pause(0.02)

  # save data every savint steps
  if ((np.mod(s,savint)==0) and (savint>0)):
    plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=' + str(s) + '.png')

  # save timeseries at diagpoints
  if (savint>0):
    for j in range(Q.shape[1]):
      add = np.zeros(Q.shape[0])
      #if (j==0):
	#add = btopo[:]-h0
      interpol[j] = func_interpolate(dgel.dofcoordinates,diagpoints,Q[:,j] + add)
    for j in range(len(diagpoints)):
      filestr = sfold + stest + '_timeseries_x=' + str(diagpoints[j]) + '.out'
      with open(filestr, "a+") as f2:
	swrite = ''
	for k in range(Q.shape[1]-1):
	  swrite = swrite + str(interpol[k,j]) + '     '
	swrite = swrite + str(interpol[k,j]) + '\n'
	f2.write(swrite)
      f2.close()

  # save data for convergence test
  if(iconvtest==1):
    if any(np.abs(timestamparr-t) <dt):
      filestrc = sfoldconv + stest + '_timestamp_N=' + str(N) + '_t=' + str(t) + '_m=' + str(m) + '.out'
      with open(filestrc, "w") as f3:
        for i in range(len(Q[:,0])):
          f3.write(str(Q[i,0]) + '     ' + str(Q[i,1]) + '     ' + str(P[i,0]) + '     ' + str(P[i,1]) + '\n')
      f3.close()

  print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s,t,dt) +
        "CFL = {0:4.2f}, ".format(CFLg) +
        "mass error: {0:+6.4e}, ".format((Qm[s]-Qm[0]).mean(axis=0)[0]) +
        "moment. error: {0:+6.4e}".format((Qm[s]-Qm[0]).mean(axis=0)[1]))

  if (np.isnan(CFLg)):
    quit()

