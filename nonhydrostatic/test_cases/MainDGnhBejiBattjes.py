"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)

This test case follows the experiment of Beji and Battjes (1993), Luth (2003) and the description in Dingemans (1994) and Stelling and Zijlema (2003).

Beji, S., and Battjes, J. (2003): Experimental investigation of wave propagation over a bar. Coastal Engineering 19(1), 151-162 (1993).
http://eng.serdarbeji.com/wp-content/uploads/2014/09/Beji-Battjes-Coastal-Eng-93.pdf

Luth HR Klopman G., K.N.: Project 13g: Kinematics of waves breaking partially on an o shore bar; ldv measurements for waves with and withoutnet onshore current. Tech. rep., Delft Hydraulics (1994)

Dingemans, M.: Comparison of computations with boussinesq-like models and laboratory measurements. memo in framework of MAST project (G8-M), Delft Hydraulics memo H1684. 12 (1994)

Stelling, G., Zijlema, M.: An accurate and efficient finite-difference algorithm for non-hydrostatic free-surface flow with application to wave propagation. International Journal for Numerical Methods in Fluids 43(1), 1-23 (2003).

"""

import matplotlib.pyplot as plt
import numpy as np
import time

from hypy1d.grid import Grid1D, generate_uniformgrid
from hypy1d.dg_element import DGReferenceElement1D, DGElement1D, DGQuadrature1D
import hypy1d.swe.riemann_solvers as eq
from hypy1d.interpolation import Vandermonde1D
from hypy1d.swe.equation import EqSWESource
from hypy1d.timestepping_r2 import RungeKuttaDG
#from hypy1d.boundary_value  import BoundaryValueDGZeroExtrap
from hypy1d.boundary_value import BoundaryValueDGInflowOutflow
#from hypy1d.flux_divergence import FluxDiv1DDGStrong as FluxDiv1DDG
from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
from hypy1d.limiter_none import Limiter

# non-hydrostatic extension
from hypy1d.elliptic.equation import EllipticEquation, AnalyticalSolution, Localnh, Globalnh
from hypy1d.factors import FactorsElliptic, Solve
from hypy1d.boundary_elliptic import LeftDirichlet as Left
from hypy1d.boundary_elliptic import RightDirichlet as Right

from test.projection_criterion import calculate_criteria, plot_criteria


# interpolation from first data points onto second data points
def func_interpolate(x1, x2, y1):
    y2 = np.zeros(len(x2))
    for i in range(len(x2)):
        m = 0.
        for j in range(len(x1) - 1):
            if ((x1[j] <= x2[i]) and (x2[i] <= x1[j + 1])):
                m = (y1[j + 1] - y1[j]) / (x1[j + 1] - x1[j])
                break
        y2[i] = m * (x2[i] - x1[j]) + y1[j]
    return y2


def multruns_input():

    value = np.loadtxt('multruns_in')

    return value


def PlotStep(Q, Qm, btopo, Grid, titletext):

    tol = 1.0e-8

    #H = btopo
    H = Q[:, 0] + btopo
    #Hm = Qm[:,0] + btopo

    ihb = 0
    #ihb = 1

    if (A > 0.):
        if (ihb == 1):
            i1 = 511
            i2 = 512
            i3 = 513
        else:
            i1 = 0
            i2 = 411
            i3 = 412
    else:
        if (ihb == 1):
            i1 = 311
            i2 = 312
            i3 = 313
        else:
            i1 = 0
            i2 = 211
            i3 = 212

    # plot data
    fig.clf()
    if (ihb == 1):
        ax5 = fig.add_subplot(i1)
        for ielt in range(gr.elength):
            intx = gr.nodecoordinates[ielt] + (
                intxre + 1.0) / 2.0 * gr.elementwidth[ielt]
            ax5.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
            ax5.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

        #ax5.plot(Grid.elementcenter, Hm[:,0], 'ro')

        ax5.set_ylabel('$h+b$')
        ax5.axis(hbrange)
        ax5.set_title(titletext)

    ax1 = fig.add_subplot(i2)
    for ielt in range(gr.elength):
        intx = gr.nodecoordinates[ielt] + (intxre +
                                           1.0) / 2.0 * gr.elementwidth[ielt]
        ax1.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 0]), 'b-')
        ax1.plot(intx[ifstlst],
                 np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 0]), 'bo')

    #ax1.plot(Grid.elementcenter, Qm[:,0], 'ro')

    ax1.set_ylabel('$h$')
    ax1.axis(hrange)
    if (ihb == 0):
        ax1.set_title(titletext)

    ax2 = fig.add_subplot(i3)
    for ielt in range(gr.elength):
        intx = gr.nodecoordinates[ielt] + (intxre +
                                           1.0) / 2.0 * gr.elementwidth[ielt]
        ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 1]), 'b-')
        ax2.plot(intx[ifstlst],
                 np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 1]), 'bo')

        #ax2.plot(Grid.elementcenter, Qm[:,1], 'ro')

    ax2.set_ylabel('$hu$')
    ax2.axis(hurange)

    if (A > 0.):
        ax3 = fig.add_subplot(i3 + 1)
        for ielt in range(gr.elength):
            intx = gr.nodecoordinates[ielt] + (
                intxre + 1.0) / 2.0 * gr.elementwidth[ielt]
            ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 2]), 'b-')
            ax3.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 2]),
                     'bo')

            #ax3.plot(Grid.elementcenter, Qm[:,2], 'ro')

        ax3.set_ylabel('$hw$')
        ax3.axis(hwrange)

        ax4 = fig.add_subplot(i3 + 2)
        for ielt in range(gr.elength):
            intx = gr.nodecoordinates[ielt] + (
                intxre + 1.0) / 2.0 * gr.elementwidth[ielt]
            ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 3]), 'b-')
            ax4.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 3]),
                     'bo')

            #ax4.plot(Grid.elementcenter, Qm[:,3], 'ro')

        ax4.set_ylabel('$pnh$')
        ax4.axis(hpnhrange)

    plt.draw()


def Initial(Grid, DGElmt):
    """
  Definition of initial conditions: experiment of beji and battjes
  """

    # bathymetry
    hmax = 0.3
    m1 = 0.05
    m2 = 0.1
    x1 = 6.
    x2 = x1 + hmax / m1
    x3 = x2 + 2.
    x4 = x3 + 3.

    Q = np.zeros((DGElmt.doflength, iiunknowns))
    b = np.zeros(DGElmt.doflength)
    hw = np.zeros(DGElmt.doflength)
    hpnh = np.zeros(DGElmt.doflength)

    #rds = 5.
    #rds2 = rds**2
    #centr = x2+1.
    #for i in range(DGElmt.doflength):
    #coo = DGElmt.dofcoordinates[i]
    #dpt = (coo - centr)**2
    #if (dpt<rds2):
    #b[i] = hmax*(0.5*(np.cos(np.sqrt(dpt)/rds*np.pi)+1.))

    for i in range(DGElmt.doflength):
        coo = DGElmt.dofcoordinates[i]

        if ((x1 <= coo) & (coo < x2)):
            b[i] = m1 * (coo - x1)
        elif ((x2 <= coo) & (coo < x3)):
            b[i] = hmax
        elif ((x3 <= coo) & (coo < x4)):
            b[i] = hmax - m2 * (coo - x3)

    h = d - b
    hu = 0.

    if (A > 0.):
        hw = 0.
        hpnh = 0.
        Q[:, 2] = hw
        Q[:, 3] = hpnh

    Q[:, 0] = h
    Q[:, 1] = hu

    return Q, b


a = 0.01
T = 2.02
#T = 0.04
h0 = 0.4
u0 = 0.0

# well-balanced test
#a = 0.0


def wleft(t):
    return 0.


def sshleft(t):
    return a * np.sin(2. * np.pi * t /
                      T)  # T=1/25=0.04s -> *csw(=sqrt(grav*h0))=Wellenlaenge
    #return a-a*np.cos(2.*np.pi*t/T) # T=1/25=0.04s -> *csw(=sqrt(grav*h0))=Wellenlaenge
    #using this, there is no unsmoothness in generation of wave


def sshrigt(t):
    return 0.0


def uleft(t):
    return 0.


def tracleft(t):
    return h0 * a * (0.5 - 0.5 * np.cos(2. * np.pi * t / T))


def tracrigt(t):
    return h0 * 0.0


diagpoints = np.array([10.5, 12.5, 13.5, 14.5, 15.7, 17.3, 19.0, 21.0])
d = h0
swnl = 1
N = 1  # polynomial interpolation order
#m = multruns_input()
m = 400  # number of grid nodes, should be same refinement as in TAM (reflev=3 dort)
xmin = 0.0  # position of left boundary
xmax = 40.0  # position of right boundary
#xmax = 60.0   # position of right boundary
Tmax = 40.0  # end time
CFL = 0.90  # CFL number
dt = 6. / (m - 1.)  # time step size, s.t. CFL=const
#grav = 100.0       # gravitational constant
grav = 9.80616  # gravitational constant
pltint = (m - 1) / 2  # steps between plots
#savint = (m-1)/2   # steps between saves
#pltint = 10        # steps between plots
savint = 0  # steps between saves or no save of plots (savint=0)
intpts = 1  # number of interpolation points within one element for visualisation

# definition non-hydrostatic set-up
A = 1.5  # flag for hydrostatic run (A=0.), or non-hydrostatic run with quadratic pressure profile (A=1.5) or linear pressure profile (A=2.)
B = 0  # variable f_d
nhnl = 1  # flag for non-linear non-hydrostatic run
nht = 2  # order of time discretization: first order (nht=1) or second order (nht=2)
iiunknowns = 4  # number of iunknowns in Q
if (A == 0.):
    B = 0
    A = 0.
    nhnl = 0
    nht = 0
    iiunknowns = 2

relt = DGReferenceElement1D(N)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

# compute initial condition
Q, btopo = Initial(gr, dgel)

# define folder to save figures and convergence results
stest = 'bejibattjes'
if (A == 0.):
    shelp = 'swe/'
elif (A == 1.5):
    shelp = 'nh2/'
elif (A == 2.):
    shelp = 'nh1/'
sfold = 'diag/' + stest + '/' + shelp
#for convergence test, just change 'sfold' to 'sfoldconv' in loop with diagpointsarr
#sfoldconv = 'diag/convergence/' + stest + '/'
sfoldconv = sfold

# connecting classes for hydrostatic code
localnh = Globalnh(gr, dgel)
eq = eq.EqSWERusanov(localnh, d, grav, swnl, nht, A, B, 1.0e-8, iiunknowns - 2)
#eq   = eq.EqSWEHLLE(d, grav, swnl, 0, A, 1.0e-8, 1)
src = EqSWESource(eq, dgel, dgqu)
#bv    = []
#bv    = [BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
#BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)]
bv = [
    BoundaryValueDGInflowOutflow(gr, gr.bnodes[0], dgel, grav, h0, u0, sshleft,
                                 iiunknowns, tracleft),
    BoundaryValueDGInflowOutflow(gr, gr.bnodes[1], dgel, grav, h0, u0, sshrigt,
                                 iiunknowns, tracrigt)
]
FD = FluxDivergence1DDG(gr, eq, src, dgel, dgqu, bv)
Lim = Limiter(gr, eq, src, dgel)
RK = RungeKuttaDG(FD, Lim, 2)
src.update(btopo)

# compute interpolation points and the mapping from the dofs for visualisation
intxre = np.linspace(-1.0, 1.0, intpts)
intpsi = np.dot(Vandermonde1D(relt.N, intxre), relt.Vinv)
ifstlst = [0, -1]
fin = np.finfo(float)

# set-up of non-hydrostatic part including local run and boundary conditions
eqell = EllipticEquation(dgel, grav, d, btopo, A, B, nhnl, nht)
fact = FactorsElliptic(gr, src, eqell, dgel, dgqu)
sol = Solve()
localnh.update_local()
# boundary condition on left and right boundary for the unknown fo the elliptic problem
#diri  = np.array([1,1])  # put (non-zero) Dirichlet boundary data here
diri = np.array(
    []
)  # in case of non-(non-zero)-Dirichlet boundary data: zero Dirichlet, Periodic, Reflection
bvell = [Left(gr, dgel, localnh), Right(gr, dgel, localnh), diri]

# compute cell mean values
Qm = np.zeros((int(Tmax / dt) + 3, gr.elength, iiunknowns))
for ielt in range(gr.elength):
    Qm[0,
       ielt] = relt.V[0, 0] * np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

shift = 0.01
hbrange = [xmin, xmax, h0 - 3 * shift, h0 + 3 * shift]
hrange = [xmin, xmax, h0 - 3 * shift, h0 + 3 * shift]
hurange = [xmin, xmax, -0.05 - shift, 0.05 + shift]
if (A > 0.):
    hwrange = [xmin, xmax, min(Q[:, 2]) - shift, max(Q[:, 2]) + shift]
    hpnhrange = [xmin, xmax, min(Q[:, 3]) - shift, max(Q[:, 3]) + shift]

# plot initial condition
# fig = plt.figure(1)
# PlotStep(Q, Qm[0], btopo, gr, 'Initial conditions')

plt.show(block=False)
if (savint > 0):
    plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=0.png')

# save timeseries at diagpoints
if (savint > 0):
    interpol = np.zeros((iiunknowns, len(diagpoints)))
    interpol[0] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                   Q[:, 0] + btopo[:] - h0)
    interpol[1] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:, 1])
    if (A > 0.):
        interpol[2] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          2])
        interpol[3] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          3])

    for j in range(len(diagpoints)):
        filestr = sfoldconv + stest + '_timeseries_x=' + str(
            diagpoints[j]) + '.out'
        with open(filestr, "w") as f2:
            if A > 0.:
                f2.write(
                    str(interpol[0, j]) + '     ' + str(interpol[1, j]) +
                    '     ' + str(interpol[2, j]) + '     ' +
                    str(interpol[3, j]) + '\n')

            if A == 0.0:
                f2.write(
                    str(interpol[0, j]) + '     ' + str(interpol[1, j]) + '\n')
            f2.close()

#while not plt.waitforbuttonpress():
#pass

# loop over time
t = 0.0
s = 0

# get previous time step for second order computation
Qold = np.copy(Q)
#Qold = np.zeros(Q.shape)
criteria = []
while t < Tmax - fin.resolution:
    if s == 1000:
        break
    criteria.append(calculate_criteria(dgel,fact, FD, Q))
    u = np.zeros(dgel.doflength)
    #mask = Q[:,0] > wettol
    #u[mask] = abs(Q[mask,1] / Q[mask,0])
    #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
    CFLg = np.max(u + np.sqrt(eq.g * Q[:, 0])) * dt / np.min(gr.elementwidth)

    Qnew, Qhelp = RK.step(Q, Qold, t, dt)

    # non-hydrostatic extension: projection method
    if (A > 0.):
        hu, dipnh = sol.Solve_sys(
            fact, Qnew, dt, t, bvell,
            Qhelp)  # solve lin. equ. sys., hu is already updated
        Qnew = sol.corr_tracer(fact, FD, Qnew, hu, dipnh, t, dt,
                               Qhelp)  # correction step
        Qnew = Lim(
            Qnew
        )  # application of limiter because correction step for water height

    Qold = Q
    Q = Qnew

    hbrange = [xmin, xmax, h0 - 3 * shift, h0 + 3 * shift]
    hrange = [xmin, xmax, h0 - 3 * shift, h0 + 3 * shift]
    hurange = [xmin, xmax, -0.05 - shift, 0.05 + shift]
    if (A > 0.):
        hwrange = [xmin, xmax, min(Q[:, 2]) - shift, max(Q[:, 2]) + shift]
        hpnhrange = [xmin, xmax, min(Q[:, 3]) - shift, max(Q[:, 3]) + shift]

    t = t + dt
    s = s + 1

    # compute cell mean values
    for ielt in range(gr.elength):
        Qm[s, ielt] = relt.V[0, 0] * np.dot(relt.Vinv[0],
                                            Q[dgel.elementdofs[ielt]])

    # plot data every pltint steps
    # if (np.mod(s, pltint) == 0):
    #     PlotStep(
    #         Q, Qm[s], btopo, gr, 't = ' + str(t) + ', step = ' + str(s) +
    #         ', CFL = ' + str(round(CFLg, 3)))
        #plt.pause(0.02)

    # save data every savint steps
    if ((np.mod(s, savint) == 0) and (savint > 0)):
        plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=' +
                    str(s) + '.png')

    # save timeseries at diagpoints
    if (savint > 0):
        interpol[0] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                       Q[:, 0] + btopo[:] - 0.4)
        interpol[1] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          1])
        if (A > 0.):
            interpol[2] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                           Q[:, 2])
            interpol[3] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                           Q[:, 3])
        for j in range(len(diagpoints)):
            filestr = sfoldconv + stest + '_timeseries_x=' + str(
                diagpoints[j]) + '.out'
            with open(filestr, "a+") as f2:
                if (A > 0.):
                    f2.write(
                        str(interpol[0, j]) + '     ' + str(interpol[1, j]) +
                        '     ' + str(interpol[2, j]) + '     ' +
                        str(interpol[3, j]) + '\n')
                if (A == 0.):
                    f2.write(
                        str(interpol[0, j]) + '     ' + str(interpol[1, j]) +
                        '\n')
        f2.close()

    print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s, t, dt) +
          "CFL = {0:4.2f}, ".format(CFLg) +
          "mass error: {0:+6.4e}, ".format((Qm[s] - Qm[0]).mean(axis=0)[0]) +
          "moment. error: {0:+6.4e}".format((Qm[s] - Qm[0]).mean(axis=0)[1]))

    if (np.isnan(CFLg)):
        quit()

plot_criteria(criteria)