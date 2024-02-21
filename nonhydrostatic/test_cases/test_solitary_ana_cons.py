"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
from math import degrees, atan

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGProlong, DGQuadrature1D, L2error, Linferror
from hypy1d.interpolation   import Vandermonde1D

import hypy1d.swe.riemann_solvers as eq
from hypy1d.swe.equation    import EqSWESource
from hypy1d.timestepping    import RungeKuttaDG
from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
from hypy1d.limiter_none    import Limiter

from hypy1d.elliptic.equation import AnalyticalSolution
from hypy1d.factors           import FactorsElliptic, Solve



def PlotStep(Q, Qm, Grid, titletext):

  tol = 1.0e-8

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(411)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  #ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(412)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

    #ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$hu$')
  ax2.axis(hurange)

  ax3 = fig.add_subplot(413)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],2]), 'b-')
    ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],2]), 'bo')

    #ax3.plot(Grid.elementcenter, Qm[:,2], 'ro')

  ax3.set_ylabel('$hw$')
  ax3.axis(hwrange)

  ax4 = fig.add_subplot(414)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],3]), 'b-')
    ax4.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],3]), 'bo')

    #ax4.plot(Grid.elementcenter, Qm[:,3], 'ro')

  ax4.set_ylabel('$hpnh$')
  ax4.axis(hpnhrange)

  #plt.draw()


iversion = 1
d = 10.
#m    = multruns_input()
swnl = 1
N    = 2      # polynomial interpolation order
m    = 21      # number of grid nodes
xmin = 0.0    # position of left boundary
xmax = 400.0    # position of right boundary
Tmax = 0.    # end time
#CFL  = 0.90   # CFL number
dt = 0.01
#dt = 0.02/(m-1.)
grav = 9.80616  # gravitational constant
#pltint = 1    # steps between plots
#savint = 100   # steps between saves
intpts = 11   # number of interpolation points within one element for visualisation
t = 0.

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

eq   = eq.EqSWERusanov(d, grav, swnl)
src   = EqSWESource(eq, dgel, dgqu)
bv    = []
#bv    = [BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
         #BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
FD    = FluxDivergence1DDG(gr, eq, src, dgel, dgqu, bv)
Lim   = Limiter(gr, eq, src, dgel)
RK    = RungeKuttaDG(FD, Lim, 2)





#def Initial(Grid, DGElmt, grav, d, xlen):
  #"""
  #Definition of initial conditions: periodic standing wave
  #"""

A    = 1.5
B    = 0.
nhnl = 1
iunknowns = 4

anasol = AnalyticalSolution(dgel, grav, d, A, B)

a = 2.
xlen = xmax-xmin
x0 = xlen/2.

K = np.sqrt(3.*a/(4.*d*d*(d+a)))
##K = np.sqrt(3.*a/(4.*d*d*(d)))
c = np.sqrt(grav*(d+a))
##c = np.sqrt(grav*d*(1.+a/(2.*d)))

Q = np.zeros((dgel.doflength,iunknowns))
#b = np.zeros(dgel.doflength)
T = np.tanh(K*(dgel.dofcoordinates-x0))


if (iversion==0):
  ssh = a/((np.cosh(K*(dgel.dofcoordinates-x0)))**2)
  h = d + ssh
  hu = c*ssh
  u = hu/h

  hw = d*c*K*ssh*T                                         # same as (w from primitive equ.) times h
  hpnh = ((d*K*c)**2)*ssh/A*(2.*(T**2)*d/h-ssh/a)          # same as (pnh from primitive equ.) times h
  # this is not the same as (pnh from primitive equ.) times h...!
  #terms = 2.*T**2*(1.-ssh/h)-ssh/a

  Q[:,2] = hw
  Q[:,3] = hpnh

  Q[:,0] = h
  Q[:,1] = hu


if(iversion==1):
  Q[:,0], Q[:,1], Q[:,2], Q[:,3] = anasol.analytical_solit_version0(0., a, x0)
  h    = Q[:,0]
  ssh  = h - d
  hu   = Q[:,1]
  hw   = Q[:,2]
  hpnh = Q[:,3]
  u    = hu/h

  #return Q, P, d, b, A, B, nhnl, T, ssh, K


def ssh_x():

  return -2.*K*ssh*T

def ssh_t():

  return -c*ssh_x()

def h_t():

  return ssh_t()

def h_x():

  return ssh_x()

def hu_x():

  return c*ssh_x()

def hu_t():

  return c*ssh_t()

def T_x():

  return ssh/a*K

def T_t():

  return -c*T_x()

def hw_x():

  return d*c*K*(ssh_x()*T+ssh*T_x())

def hw_t():

  return d*c*K*(ssh_t()*T+ssh*T_t())

#def hw_t(): #correct

  #return d*c**2*K**2*ssh*(2.*T**2-ssh/a)

def hwu_x():

  return (hw_x()*u)+(hw*u_x())

#def hwu_x(): #correct

  #return d/h*((K*ssh*c)**2)*((-2.*(T**2)*(1.+d/h))+ssh/a)

def huu_x():

  return (hu_x()*u)+(hu*u_x())

def u_x():

  return c*ssh_x()*d/(h**2)

def hpnh_x():

  return (K*c*d)**2/A*( 4.*T*T_x()*d*ssh/h +2.*((T*d/h)**2)*h_x() -2.*ssh*ssh_x()/a )



print 'check continuity equation:'           #correct
print h_t() + hu_x()

print 'check divergence constraint:'         #correct
print 2.*hw-hu*h_x()+h*hu_x()

print 'check vertical momentum equation'     #correct
print hw_t()+hwu_x()-A/h*hpnh

print 'check horizontal momentum equation:'  #correct
print hu_t()+huu_x()+grav*h*h_x()+hpnh_x()



# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
#compute cell mean values
Qm = np.zeros((int(Tmax/dt)+3,gr.elength,iunknowns))
Pm = np.zeros((int(Tmax/dt)+3,gr.elength,iunknowns))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])
#plot solution
hrange = [xmin, xmax, min(Q[:,0]), max(Q[:,0])]
hurange = [xmin, xmax, min(Q[:,1]), max(Q[:,1])]
hwrange = [xmin, xmax, min(Q[:,2]), max(Q[:,2])]
hpnhrange = [xmin, xmax, min(Q[:,3])-1., max(Q[:,3])+1.]
fig = plt.figure(1)
PlotStep(Q, Qm[0], gr, 'Initial conditions')
plt.savefig('test_solitary_ana_cons.png')



