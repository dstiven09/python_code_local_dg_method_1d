"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

This is the toy problem to solve more general elliptic equations with the Local Discontinuous Galerkin (LDG) method.

"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
from hypy1d.interpolation   import Vandermonde1D
#from hypy1d.timestepping    import RungeKuttaDG
from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap
import hypy1d.swe.riemann_solvers as eq
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
from hypy1d.limiter_none    import Limiter

# non-hydrostatic extension
import hypy1d.poisson.equation as eqell
from hypy1d.elliptic.equation import Globalnh, Localnh
from hypy1d.factors           import FactorsElliptic, Solve
from hypy1d.boundary_elliptic import LeftDirichlet as Left
from hypy1d.boundary_elliptic import RightDirichlet as Right


def multruns_input():

  value  = np.loadtxt('multruns_in')
  return value

def PlotStep(Q, Qm, Grid, titletext):

  tol = 1.0e-8

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(211)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  #ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$q$')
  #ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(212)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

    #ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$u$')
  #ax2.axis(mrange)

  plt.draw()


def Data(Grid, DGElmt):
  """
  Definition of given data
  """

  Q = np.zeros((DGElmt.doflength,4))
  b = np.zeros(DGElmt.doflength)

  Q[:,1] = np.cos(np.pi*(DGElmt.dofcoordinates+1.))*(np.pi)
  Q[:,3] = np.sin(np.pi*(DGElmt.dofcoordinates+1.))

  return Q, b

d         = 10.
swnl      = 1
stest     = 'poisson'
iconvtest = 0
ireflect  = 1

N    = 1      # polynomial interpolation order
#m = multruns_input()
m    = 9      # number of grid nodes
xmin = -1.0    # position of left boundary
xmax = 1.0    # position of right boundary
Tmax = 0.0    # end time
#CFL  = 0.90   # CFL number
dt   = 0.01  # time step size
#grav = 100.0  # gravitational constant
grav = 9.81  # gravitational constant
pltint = 1    # steps between plots
savint = 0         # steps between saves or no save of plots (savint=0)
intpts = 10   # number of interpolation points within one element for visualisation

# establish grid and DG elements
relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

eq   = eq.EqSWERusanov(d, grav)
#eq   = eq.EqSWEHLLE(d, grav, swnl, 0, A, 1.0e-8, 1)
name  = 'diag/' + stest + '/' + stest
#src   = EqSWESource(eq, dgel, dgqu)
bv    = []
#bv    = [BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
         #BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
#FD    = FluxDiv1DDG(gr, eq, src, dgel, dgqu, bv)
#Lim   = Limiter(gr, eq, src, dgel)
#RK    = RungeKuttaDG(FD, Lim, 2)

# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
fin = np.finfo(float)

# get data
anasol = eqell.AnaSol(dgel)
if (ireflect==0):
  Q = anasol.analytical_poisson()
if (ireflect==1):
  Q = anasol.analytical_poisson_refl()
btopo = np.zeros(dgel.doflength)
#Q, btopo = Data(gr, dgel)
shelp = stest
sfold = 'diag/' + stest + '/' + shelp
#for convergence test, just change 'sfold' to 'sfoldconv' in loop with timestamparr
#sfoldconv = 'diag/convergence/' + stest + '/'
sfoldconv = sfold


# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+1,gr.elength,Q.shape[1]))
col = np.array([1,3])
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

hrange = [xmin, xmax, -4., 4.]
mrange = [xmin, xmax, -4., 4.]
urange = [xmin, xmax, -1.5, 1.5]


# plot initial condition
fig = plt.figure(1)
PlotStep(Q[:,col], Q[0,col], gr, 'Initial')
plt.show
while not plt.waitforbuttonpress():
  pass
if (savint>0):
  plt.savefig(sfold + '_initial' + '_N=' + str(N) + '_m=' + str(m) + '.png')


localnh = Globalnh(gr, dgel)

# compute solution
src = eqell.EqellSource(eq, dgel, dgqu)
src.update(btopo)
eqell = eqell.Eqell(dgel, src, grav, d, btopo)
fact = FactorsElliptic(gr, src, eqell, dgel, dgqu)
sol  = Solve()

localnh.update_local()
# boundary condition on left and right boundary for the unknown fo the elliptic problem
#diri  = np.array([1,1])  # put (non-zero) Dirichlet boundary data here
diri  = np.array([])     # in case of non-(non-zero)-Dirichlet boundary data
bvell = [Left(gr, dgel, localnh), Right(gr, dgel, localnh), diri]

Q[:,1], Q[:,3] = sol.Solve_sys(fact, Q, dt, 0., bvell, Q)
# do not have to divide by d, because we are only solving the linear equation system itself

# compute cell mean values
Qm = np.zeros((int(Tmax/dt)+1,gr.elength,Q.shape[1]))
col = np.array([1,3])
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

hrange = [xmin, xmax, -4., 4.]
mrange = [xmin, xmax, -4., 4.]
urange = [xmin, xmax, -1.5, 1.5]

# plot solution
fig = plt.figure(2)
PlotStep(Q[:,col], Q[0,col], gr, 'Solution to f=sin')
plt.show
while not plt.waitforbuttonpress():
  pass
if (savint>0):
  plt.savefig(sfold + '_solution' + '_N=' + str(N) + '_m=' + str(m) + '.png')

if(iconvtest==1):
  np.savetxt(sfoldconv + stest + '_N=' + str(N) + '_m=' + str(m) + '.out', Q, fmt="%2.14f")
