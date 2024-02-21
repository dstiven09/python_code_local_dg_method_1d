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

import hypy1d.elliptic.equation as eqell
from hypy1d.elliptic.factors     import FactorsElliptic, Solve



def PlotStep(Q, Qm, P, Pm, Grid, titletext):

  tol = 1.0e-8

  # plot data
  fig.clf()
  ax1 = fig.add_subplot(411)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],0]), 'b-')
    ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],0]), 'bo')

  ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

  ax1.set_ylabel('$h$')
  ax1.axis(hrange)
  ax1.set_title(titletext)

  ax2 = fig.add_subplot(412)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'b-')
    ax2.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

    ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

  ax2.set_ylabel('$u$')
  ax2.axis(hurange)

  ax3 = fig.add_subplot(413)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax3.plot(intx, np.dot(intpsi, P[dgel.elementdofs[ielt],0]), 'b-')
    ax3.plot(intx[ifstlst], np.dot(intpsi[ifstlst], P[dgel.elementdofs[ielt],0]), 'bo')

    ax3.plot(Grid.elementcenter, Pm[:,0], 'ro')

  ax3.set_ylabel('$w$')
  ax3.axis(hwrange)
  
  ax4 = fig.add_subplot(414)
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax4.plot(intx, np.dot(intpsi, P[dgel.elementdofs[ielt],1]), 'b-')
    ax4.plot(intx[ifstlst], np.dot(intpsi[ifstlst], P[dgel.elementdofs[ielt],1]), 'bo')

    ax4.plot(Grid.elementcenter, Pm[:,1], 'ro')

  ax4.set_ylabel('$pnh$')
  ax4.axis(hpnhrange)
  
  #plt.draw()



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
nh   = 1

g = 9.80616
#d = 10.
a = 2.0
K = np.sqrt(3.*a/(4.*d*d*(d+a)))
c = np.sqrt(g*(d+a))

xlen = xmax-xmin
x0 = xlen/2.

Q = np.zeros((dgel.doflength,2))
b = np.zeros(dgel.doflength)

ssh = a/((np.cosh(K*(dgel.dofcoordinates-x0)))**2)
h = d + ssh
u = c*ssh/h

if (nh==1):
  P = np.zeros((dgel.doflength,2))
  pnh = np.zeros(dgel.doflength)
  T = np.tanh(K*(dgel.dofcoordinates-x0))
  w = d*c*K*ssh*T/h
  terms = d/h*2.*T**2-ssh/a
  pnh = (d*K*c)**2*ssh*terms/(A*h)
  P[:,0] = w
  P[:,1] = pnh

Q[:,0] = h
Q[:,1] = u

  #return Q, P, d, b, A, B, nh, nhnl, T, ssh, K


def ssh_x():

  return 2.*K*ssh*(-T)

def ssh_t():

  return -c*ssh_x()

def h_t():

  return ssh_t()

def h_x():

  return ssh_x()

def u_x():

  return c*ssh_x()*d/(h**2)

def u_t():

  return -c*u_x()

def T_x():
  
  return ssh/a*K

def T_t():
  
  return -c*T_x()

def w_x():
  
  return -d*c*(K**2)*ssh/h*(-2./h*ssh*(T**2)+2.*(T**2)-ssh/a)

def w_t():
  
  return -c*w_x()

def pnh_x():
  
  return ((d*K*c)**2)/A*(pnh_x_1()+pnh_x_2()+pnh_x_3())

def pnh_x_1():
  
  return 4.*(h_x()/(h**3)*((ssh*T)**2)-ssh*ssh_x()*(T/h)**2-T*T_x()*(ssh/h)**2)

def pnh_x_2():
  
  return 2.*(ssh_x()*(T**2)/h-ssh*h_x()*((T/h)**2)+2.*ssh/h*T*T_x())

def pnh_x_3():
  
  return -ssh/(a*h)*(2.*ssh_x()-ssh/h*h_x())




print 'check continuity equation:'
print ssh_t()+h*u_x()+u*h_x()

print 'check divergence constraint:'
print 2.*w+h*u_x()

print 'compute vertical momentum equation:'
print w_t()+u*w_x()-A/h*pnh

print 'compute horizontal momentum equation:'
print u_t()+u*u_x()+g*ssh_x()+h_x()/h*pnh+pnh_x()



# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
#compute cell mean values
Qm = np.zeros((int(Tmax/dt)+3,gr.elength,2))
Pm = np.zeros((int(Tmax/dt)+3,gr.elength,2))
for ielt in range(gr.elength):
  Qm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])
  Pm[0,ielt] = relt.V[0,0]*np.dot(relt.Vinv[0], P[dgel.elementdofs[ielt]])
#plot solution
hrange = [xmin, xmax, min(Q[:,0]), max(Q[:,0])]
hurange = [xmin, xmax, min(Q[:,1]), max(Q[:,1])]
hwrange = [xmin, xmax, min(P[:,0]), max(P[:,0])]
hpnhrange = [xmin, xmax, min(P[:,1])-1., max(P[:,1])+1.]
fig = plt.figure(1)
PlotStep(Q, Qm[0], P, Pm[0], gr, 'Initial conditions')
#plt.show(block=False)
plt.savefig('test_solitary_ana_prim.png')


