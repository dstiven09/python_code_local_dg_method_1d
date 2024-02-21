"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)

This test case follows the description of problem 1 and 4 given in
https://github.com/rjleveque/nthmp-benchmark-problems

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
from hypy1d.boundary_value import BoundaryValueDGZeroExtrap

from hypy1d.flux_divergence import FluxDivergence1DDGWeak as FluxDivergence1DDG
from hypy1d.limiter_bjsvhy1_nh import Limiter
#from hypy1d.limiter_wd  import Limiter
#from hypy1d.limiter_none  import Limiter

# non-hydrostatic extension
from hypy1d.elliptic.equation import EllipticEquation, AnalyticalSolution, Globalnh, Localnh
from hypy1d.factors_local import FactorsElliptic, Solve
from hypy1d.boundary_elliptic_local import LeftDirichlet as Left
from hypy1d.boundary_elliptic_local import RightDirichlet as Right


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


def PlotStepPart_inun(Q, btopo, titletext):

    eltl = 0
    eltr = 40
    H = Q[:, 0] + btopo
    #H = Q[:,0]
    iplotaxis = 0

    if (A > 0.):
        i1 = 511
        i2 = 512
        i3 = 513
    else:
        i1 = 311
        i2 = 312
        i3 = 313

    # plot data
    fig.clf()
    ax1 = fig.add_subplot(i1)
    #for ielt in range(eltl, eltr):
    for ielt in range(gr.elength):
        intx = gr.nodecoordinates[ielt] + (intxre +
                                           1.0) / 2.0 * gr.elementwidth[ielt]
        ax1.plot(intx, np.dot(intpsi, btopo[dgel.elementdofs[ielt]]), 'g')
        Qelt = Q[dgel.elementdofs[ielt]]
        belt = btopo[dgel.elementdofs[ielt]]
        xhmax = np.argmax(Qelt[:, 0])
        if (np.min(Qelt[:, 0]) < wettol
                and Qelt[xhmax, 0] + belt[xhmax] - np.max(belt) < wettol):
            ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'r-')
            ax1.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'ro')
        else:
            ax1.plot(intx, np.dot(intpsi, H[dgel.elementdofs[ielt]]), 'b-')
            ax1.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], H[dgel.elementdofs[ielt]]), 'bo')

    if all(H == Q[:, 0]):
        ax1.set_ylabel('$h$')
        if (iplotaxis == 1):
            ax1.axis(hrange)
    else:
        ax1.set_ylabel('$h+b$')
        #if (iplotaxis==1):
        ax1.axis(hbrange)

    ax1.set_title(titletext)

    ax2 = fig.add_subplot(i2)
    #for ielt in range(eltl, eltr):
    for ielt in range(gr.elength):
        intx = gr.nodecoordinates[ielt] + (intxre +
                                           1.0) / 2.0 * gr.elementwidth[ielt]
        Qelt = Q[dgel.elementdofs[ielt]]
        belt = btopo[dgel.elementdofs[ielt]]
        xhmax = np.argmax(Qelt[:, 0])
        if (np.min(Qelt[:, 0]) < wettol
                and Qelt[xhmax, 0] + belt[xhmax] - np.max(belt) < wettol):
            ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 1]), 'r-')
            ax2.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 1]),
                     'ro')
        else:
            ax2.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 1]), 'b-')
            ax2.plot(intx[ifstlst],
                     np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 1]),
                     'bo')

    ax2.set_ylabel('$hu$')
    if (iplotaxis == 1):
        ax2.axis(hurange)

    ax3 = fig.add_subplot(i3)
    #for ielt in range(eltl, eltr):
    for ielt in range(gr.elength):
        intx = gr.nodecoordinates[ielt] + (intxre +
                                           1.0) / 2.0 * gr.elementwidth[ielt]
        hE = np.dot(intpsi, Q[dgel.elementdofs[ielt], 0])
        mE = np.dot(intpsi, Q[dgel.elementdofs[ielt], 1])
        uE = np.zeros(hE.shape)
        #print 'hE ', hE, ielt, gr.nodecoordinates[ielt]
        mask = (hE > wettol)
        uE[mask] = mE[mask] / hE[mask]
        Qelt = Q[dgel.elementdofs[ielt]]
        belt = btopo[dgel.elementdofs[ielt]]
        xhmax = np.argmax(Qelt[:, 0])
        if (np.min(Qelt[:, 0]) < wettol
                and Qelt[xhmax, 0] + belt[xhmax] - np.max(belt) < wettol):
            ax3.plot(intx, uE, 'r-')
            ax3.plot(intx[ifstlst], uE[ifstlst], 'ro')
        else:
            ax3.plot(intx, uE, 'b-')
            ax3.plot(intx[ifstlst], uE[ifstlst], 'bo')

    ax3.set_ylabel('$u$')
    if (iplotaxis == 1):
        ax3.axis(urange)

    if (A > 0.):

        ax3 = fig.add_subplot(i3 + 1)
        for ielt in range(gr.elength):
            intx = gr.nodecoordinates[ielt] + (
                intxre + 1.0) / 2.0 * gr.elementwidth[ielt]
            Qelt = Q[dgel.elementdofs[ielt]]
            belt = btopo[dgel.elementdofs[ielt]]
            xhmax = np.argmax(Qelt[:, 0])
            if (np.min(Qelt[:, 0]) < wettol
                    and Qelt[xhmax, 0] + belt[xhmax] - np.max(belt) < wettol):
                ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 2]),
                         'r-')
                ax3.plot(intx[ifstlst],
                         np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 2]),
                         'ro')
            else:
                ax3.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 2]),
                         'b-')
                ax3.plot(intx[ifstlst],
                         np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 2]),
                         'bo')

        ax3.set_ylabel('$hw$')
        if (iplotaxis == 1):
            ax3.axis(hwrange)

        ax4 = fig.add_subplot(i3 + 2)
        for ielt in range(gr.elength):
            intx = gr.nodecoordinates[ielt] + (
                intxre + 1.0) / 2.0 * gr.elementwidth[ielt]
            Qelt = Q[dgel.elementdofs[ielt]]
            belt = btopo[dgel.elementdofs[ielt]]
            xhmax = np.argmax(Qelt[:, 0])
            if (np.min(Qelt[:, 0]) < wettol
                    and Qelt[xhmax, 0] + belt[xhmax] - np.max(belt) < wettol):
                ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 3]),
                         'r-')
                ax4.plot(intx[ifstlst],
                         np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 3]),
                         'ro')
            else:
                ax4.plot(intx, np.dot(intpsi, Q[dgel.elementdofs[ielt], 3]),
                         'b-')
                ax4.plot(intx[ifstlst],
                         np.dot(intpsi[ifstlst], Q[dgel.elementdofs[ielt], 3]),
                         'bo')

        ax4.set_ylabel('$pnh$')
        if (iplotaxis == 1):
            ax4.axis(hpnhrange)

    plt.draw()


def Initial_swe(Grid, DGElmt, anasol, iunknowns, xshift):
    """
  Definition of initial conditions: solitary wave on a simple beach
  """

    a = 0.019 * d
    if (A > 0.):
        c = np.sqrt(grav * (d + a))  # as for swe-run
    elif (A == 0.):
        c = np.sqrt(grav * d)
    c = -c  # propagation into the opposite direction
    #these are the KdV-parameters:
    K = np.sqrt(3. * a / (4. * d * d * (d)))

    beta2 = 19.85
    X0 = d * beta2
    L = np.arccosh(np.sqrt(20.)) / (K * d)
    x0 = (X0 + L) + xshift
    m = -1. / beta2

    b = np.zeros(DGElmt.doflength)
    for i in range(DGElmt.doflength):
        coo = DGElmt.dofcoordinates[i]
        if (coo < X0):
            b[i] = m * (coo - X0)

    t = 0.
    if (A == 0.):
        #h0, hu = anasol.initial_simplebeach(t, a, c, b, x0)
        h, hu = anasol.initial_simplebeach(t, a, c, b, x0)
    #if (A>0.):
    ##h0, hu, hw, pnh = anasol.initial_simplebeach(t, a, c, b, x0)
    #h0, hu, hw, pnh = anasol.initial_simplebeach(t, a, c, b, x0)
    #ssh = h0 - d + b
    #h = d - b + ssh
    #if (A==0.):
    # as for swe-run
    #h = h0
    #ssh = h0 - d
    #hu  = h*2.*(np.sqrt(grav*(d+ssh)) - c)

    for i in range(DGElmt.doflength):
        if (h[i] < wettol):
            h[i] = 0.
            hu[i] = 0.
            #pnh[i] = 0.
        #else:
        #pnh = hpnh/h

    Q = np.zeros((DGElmt.doflength, iunknowns))
    Q[:, 0] = h
    Q[:, 1] = hu
    if (A > 0.):
        Q[:, 2] = hw
        Q[:, 3] = pnh

    return Q, b, c, a, x0


def Initial_nh(Grid, DGElmt, anasol, iunknowns, xshift):
    """
  Definition of initial conditions: solitary wave on a simple beach
  """

    a = 0.019 * d
    if (A > 0.):
        c = np.sqrt(grav * (d + a))  # as for swe-run
    elif (A == 0.):
        c = np.sqrt(grav * d)
    c = -c  # propagation into the opposite direction
    #these are the KdV-parameters:
    K = np.sqrt(3. * a / (4. * d * d * (d)))

    beta2 = 19.85
    X0 = d * beta2
    L = np.arccosh(np.sqrt(20.)) / (K * d)
    x0 = (X0 + L) + xshift
    m = -1. / beta2

    b = np.zeros(DGElmt.doflength)
    for i in range(DGElmt.doflength):
        coo = DGElmt.dofcoordinates[i]
        if (coo < X0):
            b[i] = m * (coo - X0)

    t = 0.
    #if (A==0.):
    ##h0, hu = anasol.initial_simplebeach(t, a, c, b, x0)
    #h0, hu = anasol.initial_simplebeach(t, a, c, b, x0)
    if (A > 0.):
        #h0, hu, hw, pnh = anasol.initial_simplebeach(t, a, c, b, x0)
        h, hu, hw, pnh = anasol.initial_simplebeach(t, a, c, b, x0)
    #ssh = h0 - d + b
    #h = d - b + ssh
    #if (A==0.):
    # as for swe-run
    #h = h0
    #ssh = h0 - d
    #hu  = h*2.*(np.sqrt(grav*(d+ssh)) - c)

    for i in range(DGElmt.doflength):
        if (h[i] < wettol):
            h[i] = 0.
            hu[i] = 0.
            hw[i] = 0.
            pnh[i] = 0.

    Q = np.zeros((DGElmt.doflength, iunknowns))
    Q[:, 0] = h
    Q[:, 1] = hu
    if (A > 0.):
        Q[:, 2] = hw
        Q[:, 3] = pnh

    return Q, b, c, a, x0


d = 1.
swnl = 1
N = 1  # polynomial interpolation order
m = 400  # number of grid nodes, same refinement as in TAM (reflev=3 dort)
#m = multruns_input()
xmin = -30.0  # position of left boundary
xmax = 70.0  # position of right boundary
Tmax = 40.0  # end time
CFL = 0.90  # CFL number
dt = 8. / (m - 1.)  # time step size, s.t. CFL=const
#grav = 100.0    # gravitational constant
grav = 9.80616  # gravitational constant
pltint = (m - 1) / 8.  # steps between plots
savint = (m - 1) / 8.  # steps between saves
#pltint = 1       # steps between plots
savint = 0  # steps between saves or no save of plots (savint=0)
intpts = 3  # number of interpolation points within one element for visualisation
wettol = 1e-8

# definition non-hydrostatic set-up
ilocal = 0  # flag for local run
nhcrit = 3. + wettol  # criterion definition for local run (see hypy1d/factors_local.py in routine local_arrays)
A = 1.5  # flag for hydrostatic run (A=0.), or non-hydrostatic run with quadratic pressure profile (A=1.5) or linear pressure profile (A=2.)
B = 0.  # variable fd
nhnl = 1  # flag for non-linear non-hydrostatic run
nht = 1  # order of time discretization: first order (nht=1) or second order (nht=2)
iunknowns = 4  # number of iunknowns in Q
if (A == 0.):
    A = 0.
    B = 0.
    nhnl = 0
    nht = 0
    iunknowns = 2

tau = np.sqrt(d / grav)
timestamparrwrite = np.array(
    [25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
timestamparr = timestamparrwrite * tau
diagpoints = np.array([0.25, 9.95]) / d

relt = DGReferenceElement1D(N)
#ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=True)
ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m, periodic=False)
gr = Grid1D(ndcoo, eltnd, ndels)
dgel = DGElement1D(gr, relt)
dgqu = DGQuadrature1D(relt, N)

# compute initial condition
anasol = AnalyticalSolution(dgel, grav, d, A, B)
#xshift = 15.
xshift = 0.
if (A == 0.):
    Q, btopo, c, a, x0 = Initial_swe(gr, dgel, anasol, iunknowns, xshift)
if (A > 0.):
    Q, btopo, c, a, x0 = Initial_nh(gr, dgel, anasol, iunknowns, xshift)
tshift = (xshift) / c
timestamparr = timestamparr - tshift

# define folder to save figures and convergence results
stest = 'simplebeach'
if (A == 0.):
    shelp = 'swe/m=' + str(int(m)) + '/'
if (A == 1.5):
    shelp = 'nh2/m=' + str(int(m)) + '/'
if (A == 2.):
    shelp = 'nh1/m=' + str(int(m)) + '/'
sfold = 'diag/' + stest + '/' + shelp

# connecting classes for hydrostatic code
if (ilocal == 1):
    localnh = Localnh(gr, dgel)
if (ilocal == 0):
    localnh = Globalnh(gr, dgel)
eq = eq.EqSWERusanov(localnh, d, grav, swnl, nht, A, B, wettol, iunknowns - 2)
#eq   = eq.EqSWEHLLE(d, grav, swnl, wettol, iunknowns-2)
src = EqSWESource(eq, dgel, dgqu)
#bv    = []
bv = [
    BoundaryValueDGZeroExtrap(gr, gr.bnodes[0], dgel),
    BoundaryValueDGZeroExtrap(gr, gr.bnodes[1], dgel)
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
fact = FactorsElliptic(gr, src, eqell, dgel, dgqu, localnh)
sol = Solve()
if (ilocal == 1):
    lnh, lelmt, dofsnh = sol.local_arrays(fact, FD, Q, Q, dgel, 0.,
                                          nhcrit)  # for local run
    localnh.update_local(lnh, lelmt, dofsnh)
if (ilocal == 0):
    localnh.update_local()
#diri  = np.array([1,1])  # put (non-zero) Dirichlet boundary data here
diri = np.array(
    []
)  # in case of non-(non-zero)-Dirichlet boundary data: zero Dirichlet, Periodic, Reflection
bvell = [Left(gr, dgel, localnh), Right(gr, dgel, localnh), diri]

# compute cell mean values
Qm = np.zeros((int(Tmax / dt) + 3, gr.elength, iunknowns))
for ielt in range(gr.elength):
    Qm[0,
       ielt] = relt.V[0, 0] * np.dot(relt.Vinv[0], Q[dgel.elementdofs[ielt]])

shift = 0.01
pltmin = xmin
pltmax = xmax
hrange = [pltmin, pltmax, min(Q[:, 0]), max(Q[:, 0])]
hbrange = [pltmin, pltmax, d - a, d + 0.1]
hurange = [pltmin, pltmax, -a * 4., a * 4.]
urange = [pltmin, pltmax, -a, 3.5]
if (A > 0.):
    hwrange = [pltmin, pltmax, min(Q[:, 2]) - shift, max(Q[:, 2]) + shift]
    hpnhrange = [pltmin, pltmax, -0.0015, 0.0005]

# plot initial condition
fig = plt.figure(1)
PlotStepPart_inun(Q, btopo, 'Initial conditions')

plt.show(block=False)
plt.draw()
if (savint > 0):
    plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=0.png')

#while not plt.waitforbuttonpress():
#pass

# loop over time
t = 0.0
s = 0

# save timeseries at diagpoints
if (savint > 0):
    interpol = np.zeros((iunknowns, len(diagpoints)))
    interpol[0] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                   Q[:, 0] - d + btopo)
    interpol[1] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:, 1])
    if (iunknowns == 4):
        interpol[2] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          2])
        interpol[3] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          3])
    for j in range(len(diagpoints)):
        filestr = sfold + stest + '_timeseries_x=' + str(
            diagpoints[j]) + '.out'
        with open(filestr, "w") as f2:
            if (A == 0.):
                swrite = str(t) + '     ' + str(
                    interpol[0, j]) + '     ' + str(interpol[1, j]) + '\n'
            if (A > 0.):
                swrite = str(t) + '     ' + str(interpol[
                    0, j]) + '     ' + str(interpol[1, j]) + '     ' + str(
                        interpol[2, j]) + '     ' + str(interpol[3, j]) + '\n'
                f2.write(swrite)
        f2.close()

# compute maximum runup
maxrunup = 0.
maxlocation = 0.
for i in range(len(Q[:, 0]) - 2):
    if ((Q[i, 0] > 0.) and (Q[i + 1, 0] == 0.) and (Q[i + 2, 0] == 0.)):
        if (Q[i, 0] + btopo[i] - d > maxrunup):
            maxrunup = Q[i, 0] + btopo[i] - d
            maxlocation = dgel.dofcoordinates[i]

# get previous time step for second order computation
Qold = np.zeros(Q.shape)
if (A == 0.):
    Qold[:, 0], Qold[:, 1] = anasol.initial_simplebeach(-dt, a, c, btopo, x0)
if (A > 0.):
    Qold[:, 0], Qold[:, 1], Qold[:, 2], Qold[:,
                                             3] = anasol.initial_simplebeach(
                                                 -dt, a, c, btopo, x0)

Q0 = Q

while t < Tmax - fin.resolution:
    u = np.zeros(dgel.doflength)
    mask = Q[:, 0] > wettol
    u[mask] = abs(Q[mask, 1] / Q[mask, 0])
    #dt = CFL*np.min(gr.elementwidth) / np.max(u+np.sqrt(eq.g*Q[:,0]))
    CFLg = np.max(u[:] + np.sqrt(eq.g * Q[:, 0])) * dt / np.min(
        gr.elementwidth)
    CFLu = np.max(u) * dt / np.min(gr.elementwidth)

    Qnew, Qhelp = RK.step(Q, Qold, t, dt)

    # non-hydrostatic extension: projection method
    if (A > 0.):

        if (ilocal == 1):
            lnh, lelmt, dofsnh = sol.local_arrays(fact, FD, Qnew, Q0, dgel, t,
                                                  nhcrit)  # for local run
            localnh.update_local(lnh, lelmt, dofsnh)

        if (fact.Local.doflenloc > 0):
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

    t = t + dt
    s = s + 1

    # compute cell mean values
    for ielt in range(gr.elength):
        Qm[s, ielt] = relt.V[0, 0] * np.dot(relt.Vinv[0],
                                            Q[dgel.elementdofs[ielt]])

    # plot data every pltint steps
    if ((np.mod(s, pltint) == 0)):
        PlotStepPart_inun(
            Q, btopo, 't = ' + str(t) + ', step = ' + str(s) + ', CFL = ' +
            str(round(CFLg, 3)))
        #plt.pause(0.02)

    # save data every savint steps
    if ((np.mod(s, savint) == 0) and (savint > 0)):
        plt.savefig(sfold + stest + '_N=' + str(N) + '_dt=' + str(dt) + '_s=' +
                    str(s) + '.png')

    # save timestamps
    for i in range(len(timestamparr)):
        if (np.abs(timestamparr[i] - t) < dt):
            #print timestamparr[i], t
            filestr = sfold + stest + '_timestamp_N=' + str(N) + '_t=' + str(
                timestamparrwrite[i]) + '_m=' + str(m) + '.out'
            with open(filestr, "w") as f1:
                for i in range(Q.shape[0]):
                    swrite = ''
                    for k in range(Q.shape[1] - 1):
                        swrite = swrite + str(Q[i, k]) + '     '
                    swrite = swrite + str(Q[i, Q.shape[1] - 1]) + '\n'
                    f1.write(swrite)

    # save timeseries at diagpoints
    if (savint > 0):
        interpol[0] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                       Q[:, 0] - d + btopo)
        interpol[1] = func_interpolate(dgel.dofcoordinates, diagpoints, Q[:,
                                                                          1])
        if (iunknowns > 2):
            interpol[2] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                           Q[:, 2])
            interpol[3] = func_interpolate(dgel.dofcoordinates, diagpoints,
                                           Q[:, 3])
        for j in range(len(diagpoints)):
            filestr = sfold + stest + '_timeseries_x=' + str(
                diagpoints[j]) + '.out'
            with open(filestr, "a+") as f2:
                if (A == 0.):
                    swrite = str(t) + '     ' + str(
                        interpol[0, j]) + '     ' + str(interpol[1, j]) + '\n'
                if (A > 0.):
                    swrite = str(t) + '     ' + str(
                        interpol[0, j]) + '     ' + str(
                            interpol[1, j]) + '     ' + str(
                                interpol[2, j]) + '     ' + str(
                                    interpol[3, j]) + '\n'
                f2.write(swrite)
            f2.close()

    # compute maximum runup
    for i in range(len(Q[:, 0]) - 2):
        if ((Q[i, 0] > 0.) and (Q[i + 1, 0] == 0.) and (Q[i + 2, 0] == 0.)):
            if (Q[i, 0] + btopo[i] - d > maxrunup):
                maxrunup = Q[i, 0] + btopo[i] - d
                maxlocation = dgel.dofcoordinates[i]

    print("step: {0:4d}, time: {1:8.3f}, dt = {2:5.3f}, ".format(s, t, dt) +
          "CFLg = {0:4.2f}, CFLu = {1:4.2f}, ".format(CFLg, CFLu) +
          "mass error: {0:+6.4e}, ".format((Qm[s] - Qm[0]).mean(axis=0)[0]) +
          "moment. error: {0:+6.4e}".format((Qm[s] - Qm[0]).mean(axis=0)[1]))

    if (np.isnan(CFLg) or np.isnan(CFLg)):
        quit()

filemaxrunup = sfold + 'simple_maxrunup.out'
with open(filemaxrunup, "w") as f1:
    f1.write('maximum runup:                   ' + str(maxrunup) + '\n' +
             'x-coordinate at maximum runup:   ' + str(maxlocation))
f1.close()
