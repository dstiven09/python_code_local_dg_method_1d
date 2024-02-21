"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGQuadrature1D
from hypy1d.interpolation   import Vandermonde1D

def Initial(Grid, DGElmt, grav):

  Q = np.zeros((DGElmt.doflength,2))
  b = np.zeros(DGElmt.doflength)

  b      = np.maximum(0.0, 0.025*(DGElmt.dofcoordinates - 15.72))
  Q[:,0] = np.maximum(0.0, 0.3-b)
  return Q, b

def PlotStepH(Q, btopo, *args, **kwargs):

  Qr = kwargs.get('Qref', None)

  eltl = 0
  eltr = 40

  # plot data
  fig.subplots_adjust(bottom=0.17,left=0.15)
  fig.clf()
  ax1 = fig.add_subplot(111)
  H = Q[:,0] + btopo

  if (Qr != None):
    #p2, = ax1.plot(Qr[:,0], Qr[:,1]+5000.0, '--', linewidth=2.0, color = '0.6', label="ref. solution")
    p2, = ax1.plot(Qr[:,0], Qr[:,1]+5000.0, '--', linewidth=2.0, color = 'b', label="ref. solution")
  #for ielt in range(eltl, eltr):
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), linewidth=1.0, color = '0.6')
    #p1, = ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'k', linewidth=0.5, label="DG solution")
    p1, = ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'r', linewidth=0.5, label="DG solution")
    #ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

  ax1.axis(hrange)
  #ax1.legend([p1, p2], ["DG solution", "ref. solution"])
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$h+b$')

  plt.draw()


def PlotStepM(Q):

  eltl = 0
  eltr = 40

  # plot data
  fig.subplots_adjust(bottom=0.17,left=0.15)
  fig.clf()
  ax1 = fig.add_subplot(111)

  #for ielt in range(eltl, eltr):
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    #p1, = ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'k', linewidth=0.5, label="DG solution")
    p1, = ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt],1]), 'r', linewidth=0.5, label="DG solution")
    #ax1.plot(intx[ifstlst], np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt],1]), 'bo')

  ax1.axis(mrange)
  #ax1.legend([p1, p2], ["DG solution", "ref. solution"])
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$hu$')

  plt.draw()


def PlotStepU(Q):

  eltl = 0
  eltr = 40

  # plot data
  fig.subplots_adjust(bottom=0.17,left=0.15)
  fig.clf()
  ax1 = fig.add_subplot(111)

  #for ielt in range(eltl, eltr):
  for ielt in range(gr.elength):
    intx = gr.nodecoordinates[ielt]+(intxre+1.0)/2.0*gr.elementwidth[ielt]
    hE = np.dot(intpsi, Q[dgel.elementdofs[ielt],0])
    mE = np.dot(intpsi, Q[dgel.elementdofs[ielt],1])
    uE       = np.zeros(hE.shape)
    mask     = (hE > wettol)
    uE[mask] = mE[mask]/hE[mask]
    #p1, = ax1.plot(intx, uE, 'k', linewidth=0.5, label="DG solution")
    p1, = ax1.plot(intx, uE, 'r', linewidth=0.5, label="DG solution")

  ax1.axis(urange)
  #ax1.legend([p1, p2], ["DG solution", "ref. solution"])
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$u$')

  plt.draw()


N    = 1       # polynomial interpolation order
m    = 129     # number of grid nodes
xmin = 0.0     # position of left boundary
xmax = 32.0    # position of right boundary
Tmax = 80      # end time
dt   = 0.025   # time step size
grav = 9.81    # gravitational constant
wettol = 1e-8  # wet tolerance
intpts = 7     # number of interpolation points within one element for visualisation

expmt = 'WF2a_DG_FDSWEWeak_LimBJSVhy1_tol1e-08_m129_dt0_025'

hrange = [0, xmax, 0.26, 0.35]
mrange = [0, xmax, -0.025, 0.025]
urange = [0, xmax, -0.2, 0.2]

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr   = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)

# compute interpolation points and the mapping from the dofs for visualisation
intxre  = np.linspace(-1.0, 1.0, intpts)
intpsi  = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
smax    = int(round(Tmax/dt))

# compute initial condition
Q, btopo = Initial(gr, dgel, grav)
QQ = np.load('WaveFlumeDraehne/QQ_'+expmt+'.npy')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plot snapshots
fig = plt.figure(1)
PlotStepH(QQ[int(0/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t00_h.pdf')

fig = plt.figure(2)
PlotStepM(QQ[int(0/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t00_m.pdf')

fig = plt.figure(3)
PlotStepH(QQ[int(30/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t30_h.pdf')

fig = plt.figure(4)
PlotStepM(QQ[int(30/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t30_m.pdf')

fig = plt.figure(5)
PlotStepH(QQ[int(40/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t40_h.pdf')

fig = plt.figure(6)
PlotStepM(QQ[int(40/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t40_m.pdf')

fig = plt.figure(7)
PlotStepH(QQ[int(50/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t50_h.pdf')

fig = plt.figure(8)
PlotStepM(QQ[int(50/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t50_m.pdf')

fig = plt.figure(9)
PlotStepH(QQ[int(60/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t60_h.pdf')

fig = plt.figure(10)
PlotStepM(QQ[int(60/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t60_m.pdf')

fig = plt.figure(11)
PlotStepH(QQ[int(70/dt)], btopo)
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t70_h.pdf')

fig = plt.figure(12)
PlotStepM(QQ[int(70/dt)])
plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_t70_m.pdf')


measur = np.loadtxt('WaveFlumeDraehne/Wellenposition_angepasst.txt',skiprows=2)
ts     = [-0.26, -1.13, -1.22, -1.11, -1.11, 3.56]
icoast = np.zeros(smax+1)

for i in range(smax+1):
  icoast[i] = np.where(QQ[i,:,0]<1e-5)[0][0]

fig = plt.figure(13)
fig.subplots_adjust(bottom=0.17,left=0.15)
fig.clf()
ax1 = fig.add_subplot(111)
plt.plot(np.linspace(0.0, Tmax,  int(Tmax/dt)+1), btopo[np.int32(icoast)], 'b', linewidth=0.5)
#ax1.axis([0, Tmax, 4990.0, 5005.0])
ax1.set_title('runup height')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$[m]$')

plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_runuphgt.pdf')

fig = plt.figure(14)
fig.subplots_adjust(bottom=0.17,left=0.15)
fig.clf()
ax1 = fig.add_subplot(111)
plt.plot(np.linspace(0.0, Tmax,  int(Tmax/dt)+1), dgel.dofcoordinates[np.int32(icoast)], 'b', linewidth=1.0)
plt.plot(measur[:66, 0]+35.0-ts[0], measur[:66, 3]+28.4, linewidth=0.5)
plt.plot(measur[:49, 4]+35.0-ts[1], measur[:49, 7]+28.4, linewidth=0.5)
plt.plot(measur[:58, 8]+35.0-ts[2], measur[:58,11]+28.4, linewidth=0.5)
plt.plot(measur[:56,12]+35.0-ts[3], measur[:56,15]+28.4, linewidth=0.5)
plt.plot(measur[:64,16]+35.0-ts[4], measur[:64,19]+28.4, linewidth=0.5)
plt.plot(measur[:  ,20]+35.0-ts[5], measur[:  ,23]+28.4, linewidth=0.5)
ax1.axis([20, 80, 26.0, 30.5])
#ax1.set_title('coastline position')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$[m]$')

plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_coastline.pdf')


data = np.loadtxt('WaveFlumeDraehne/Zeitserien_T30_H2a.txt',skiprows=1)

findc0  = dgel.dofcoordinates[dgel.elementdofs[:,0]] <= 0.0
findc1  = dgel.dofcoordinates[dgel.elementdofs[:,1]] >  0.0
findc   = findc0 & findc1
dofcoo  = dgel.dofcoordinates[dgel.elementdofs[findc]][0]
dofxrel = (0.0-dofcoo[0])/(dofcoo[1]-dofcoo[0])*2.0-1.0
psi     = np.dot(Vandermonde1D(relt.N, dofxrel), relt.Vinv)

hwp = np.zeros(smax+1)
for i in range(smax+1):
  hwp[i] = np.dot(psi, QQ[i,dgel.elementdofs[findc][0],0])

fig = plt.figure(15)
fig.subplots_adjust(bottom=0.17,left=0.15)
fig.clf()
ax1 = fig.add_subplot(111)
plt.plot(np.linspace(0.0, Tmax,  int(Tmax/dt)+1), hwp, 'b')
plt.plot(data[:,0]-1.0, 0.3 + (data[:,3]-data[100,3])/100.0, 'r')
ax1.axis([0, 80, 0.288, 0.313])
#ax1.set_title('fluid depth WP1')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$[m]$')

plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_h_WP1.pdf')


findc0  = dgel.dofcoordinates[dgel.elementdofs[:,0]] <= 9.56
findc1  = dgel.dofcoordinates[dgel.elementdofs[:,1]] >  9.56
findc   = findc0 & findc1
dofcoo  = dgel.dofcoordinates[dgel.elementdofs[findc]][0]
dofxrel = (9.56-dofcoo[0])/(dofcoo[1]-dofcoo[0])*2.0-1.0
psi     = np.dot(Vandermonde1D(relt.N, dofxrel), relt.Vinv)

hwp = np.zeros(smax+1)
for i in range(smax+1):
  hwp[i] = np.dot(psi, QQ[i,dgel.elementdofs[findc][0],0])

fig = plt.figure(16)
fig.subplots_adjust(bottom=0.17,left=0.15)
fig.clf()
ax1 = fig.add_subplot(111)
plt.plot(np.linspace(0.0, Tmax,  int(Tmax/dt)+1), hwp, 'b')
plt.plot(data[:,0]-1.0, 0.3 + (data[:,4]-data[100,4])/100.0, 'r')
ax1.axis([0, 80, 0.288, 0.313])
#ax1.set_title('fluid depth WP2')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$[m]$')

plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_h_WP2.pdf')


findc0 = dgel.dofcoordinates[dgel.elementdofs[:,0]] <= 15.72
findc1 = dgel.dofcoordinates[dgel.elementdofs[:,1]] >  15.72
findc  = findc0 & findc1
dofcoo  = dgel.dofcoordinates[dgel.elementdofs[findc]][0]
dofxrel = (15.72-dofcoo[0])/(dofcoo[1]-dofcoo[0])*2.0-1.0
psi     = np.dot(Vandermonde1D(relt.N, dofxrel), relt.Vinv)

hwp = np.zeros(smax+1)
for i in range(smax+1):
  hwp[i] = np.dot(psi, QQ[i,dgel.elementdofs[findc][0],0])

fig =findc0 = dgel.dofcoordinates[dgel.elementdofs[:,0]] <= 15.72
findc1 = dgel.dofcoordinates[dgel.elementdofs[:,1]] >  15.72
findc  = findc0 & findc1
dofcoo  = dgel.dofcoordinates[dgel.elementdofs[findc]][0]
dofxrel = (15.72-dofcoo[0])/(dofcoo[1]-dofcoo[0])*2.0-1.0
psi     = np.dot(Vandermonde1D(relt.N, dofxrel), relt.Vinv)

hwp = np.zeros(smax+1)
for i in range(smax+1):
  hwp[i] = np.dot(psi, QQ[i,dgel.elementdofs[findc][0],0])

fig = plt.figure(17)
fig.subplots_adjust(bottom=0.17,left=0.15)
fig.clf()
ax1 = fig.add_subplot(111)
plt.plot(np.linspace(0.0, Tmax,  int(Tmax/dt)+1), hwp, 'b')
plt.plot(data[:,0]-1.0, 0.3 + (data[:,5]-data[100,5])/100.0, 'r')
ax1.axis([0, 80, 0.288, 0.313])
#ax1.set_title('fluid depth WP3')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$[m]$')

plt.show(block=False)
plt.draw()

#fig.set_size_inches(4.5,3)
fig.set_size_inches(4.5,2.5)
fig.savefig('WaveFlumeDraehne/WaveFlume_'+expmt+'_h_WP3.pdf')
