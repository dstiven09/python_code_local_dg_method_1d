"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

functionality specific to an elliptic equation, using the Local DG method (LDG)
"""

import numpy as np


class Localnh:
    """
    class for local non-hydrostaic run
    """

    def __init__(self, Gr, DGEl):
        self.doflenglo = DGEl.doflength
        self.elmtlenglo = Gr.elength
        self.dofglo = np.arange(DGEl.doflength)
        self.elmtglo = np.arange(Gr.elength)
        self.eltdofsglo = DGEl.elementdofs

    def update_local(self, lnh, lelmt, dofsnh):
        self.doflenloc = len(lnh)
        self.elmtlenloc = len(lelmt)
        self.dofloc = lnh
        self.elmtloc = lelmt
        self.eltdofsloc = dofsnh


class Globalnh:
    """
    class for global non-hydrostaic run
    """

    def __init__(self, Gr, DGEl):
        self.doflenglo = DGEl.doflength
        self.elmtlenglo = Gr.elength
        self.dofglo = np.arange(DGEl.doflength)
        self.elmtglo = np.arange(Gr.elength)
        self.eltdofsglo = DGEl.elementdofs

    def update_local(self):
        self.doflenloc = self.doflenglo
        self.elmtlenloc = self.elmtlenglo
        self.dofloc = self.dofglo
        self.elmtloc = self.elmtglo
        self.eltdofsloc = self.eltdofsglo


class EllipticEquation:
    """
    elliptic equations object

    explanantion of factors' definition using the example of the 1D Poisson equation:
    elliptic equation:                   (var2)_xx = f
    first order sys of equ.:             (var2)_x  = (var1)
                                         (var1)_x  = f
    more general first equation:  f1_var2*(var2)_x + s1_var1*(var1) + s1_var2*(var2) = s1_c
    more general second equation: f2_var1*(var1)_x + s2_var1*(var1) + s2_var2*(var2) = s2_c
    So the factors are defined as follows:
    f1_var2 = 1, s1_var1 = -1, s1_var2 = 0, s1_c = 0,
    f2_var1 = 1, s2_var1 = 0,  s2_var2 = 0, s2_c = f.
    """

    def __init__(self, DGElmt, g, d, btopo, A, B, nhnl, nht=1, wettol=1.0e-8):
        self.DGEl = DGElmt
        self.g = g
        self.d = d
        self.unknowns = 2
        self.wettol = wettol
        self.nhA = A
        self.nhB = B
        self.nht = nht
        self.nhnl = nhnl
        self.btopo = btopo

    def u(self, qi):
        """
        compute velocity from state vector qi = (q, u), taking care of dry states
        """

        ui = np.zeros(len(qi[:, 0]))

        for i in range(len(qi[:, 0])):
            if (qi[i, 0] < self.wettol):
                ui[i] = 0.
            else:
                ui[i] = qi[i, 1]/qi[i, 0]

        return ui

    def fact_f1_var2(self, qi):
        """
        factor of source term in first equation in front of second variable
        """

        return np.ones(len(qi[:, 0]))

    def fact_s1_var1(self, qi, bi, dt):
        """
        factor of source term in first equation in front of second variable
        """

        if (self.nhnl == 0):
            res = np.ones(len(qi[:, 0]))/dt/(self.d-bi)
        else:
            res = 1./dt/qi[:, 0]

        return res

    def fact_s1_var2(self, qi, dhi, dbi, bi):
        """
        factor of source term in first equation in front of second variable
        """

        if (self.nhnl == 0):
            res = (-dbi+self.nhA*dbi)/(self.d-bi)
        else:
            res = (dhi+self.nhA*dbi)/qi[:, 0]

        return res

    def fact_s1_c(self, qi, bi, dt):
        """
        source term in first equation on right hand side
        """

        if (self.nhnl == 0):
            res = qi[:, 1]/dt/(self.d-bi)
        else:
            res = qi[:, 1]/dt/qi[:, 0]

        return res

    def fact_s1_c_add(self, qi, qiold, dqiold, bi, dbi):
        """
        source term in first equation on right hand side
        """

        if (self.nhnl == 0):
            res = (dqiold[:, 3]*(self.d-bi)-dbi*qiold[:, 3] +
                   self.nhA*qiold[:, 3]*dbi)/(self.d-bi)
        else:
            res = (dqiold[:, 0]*qiold[:, 3]+qiold[:, 0] *
                   dqiold[:, 3]+self.nhA*qiold[:, 3]*dbi)/qi[:, 0]

        return res

    def fact_f2_var1(self, qi):
        """
        factor of source term in second equation in front of gradient of first variable
        """

        return np.ones(len(qi[:, 0]))

    def fact_s2_var1(self, qi, dhi, bi, dbi, dt):
        """
        factor of source term in second equation in front of gradient of first variable
        """

        if (self.nhnl == 0):
            res = -dbi/(self.d-bi)
        else:
            res = -(dhi+2.*dbi)/qi[:, 0]

        return res

    def fact_s2_var2(self, qi, bi, dt):
        """
        factor of source term in second equation in front of gradient of first variable
        """

        if (self.nhnl == 0):
            res = 2.*dt*self.nhA/(self.d-bi)
        else:
            res = 2.*dt*self.nhA/qi[:, 0]

        return res

    def fact_s2_c(self, qi, bi, ielmt):
        """
        source term in second equation on right hand side
        """

        if (self.nhnl == 0):
            res = -2.*qi[:, 2]/(self.d-bi)
        else:
            res = -2.*qi[:, 2]/qi[:, 0]

        return res

    def fact_s2_c_add(self, qi, qiold, bi, dt):
        """
        source term in first equation on right hand side
        """

        if (self.nhnl == 0):
            res = 2.*dt*self.nhA*qiold[:, 3]/(self.d-bi)
        else:
            res = 2.*dt*self.nhA*qiold[:, 3]/qi[:, 0]

        return res

    def corr_hw(self, qi, pnh, dt, qiold):
        """
        correction term for vertical velocity hw
        """

        res = qi[:, 2]+dt*self.nhA*pnh

        if (self.nht == 2):
            res = res - dt*self.nhA*qiold[:, 3]

        return res

    def corr_hflx(self, qi, hu, dt):
        """
        correction term for water depth h
        """

        return dt*(hu-qi[:, 1])

    def hwpredict_flx(self, qi, hw, dt):
        """
        flux to compute the predicted value for vertical velocity hw
        """

        ui = self.u(qi)

        return dt*hw*ui

    def B(self, qi, dbi, ddbi, dxi):
        """
        handling of term B to get approximate behaviour to Green-Naghdi equations
        """

        ui = self.u(qi)
        if (self.nhB == 1):
            B = 0.25*qi[:, 0]*(-self.g*dbi*dxi+ui*ddbi*ui)
        else:
            B = 0.

        return B


class AnalyticalSolution:
    """
    analytical solutions to dispersive equation set for different test cases
    """

    def __init__(self, DGElmt, g, d, A, B):
        self.DGEl = DGElmt
        self.g = g
        self.d = d
        self.nhA = A
        self.nhB = B

    # interpolation from first data points onto second data points
    def func_interpolate(self, x1, x2, y1):
        y2 = np.zeros(len(x2))
        for i in range(len(x2)):
            m = 0.
            for j in range(len(x1)-1):
                if ((x1[j] <= x2[i]) and (x2[i] <= x1[j+1])):
                    m = (y1[j+1]-y1[j])/(x1[j+1]-x1[j])
                    break
                y2[i] = m*(x2[i]-x1[j])+y1[j]
            return y2

    def analytical_stand_Diri(self, t, a):
        """
        analytic solution of the non-hydrostatic pressure and the horizontal velocity hu for standing wave test case with periodic boundaries as well as with zero Dirichlet boundaries
        """

        x = self.DGEl.dofcoordinates
        A = self.nhA
        l = max(x) - min(x)
        d = self.d
        g = self.g
        k = 2.*np.pi/l
        c = np.sqrt(g*d)
        if (A > 0.):
            c = c/np.sqrt(1.+(k*d)**2/(2.*A))

        h = d+a*np.sin(k*x)*np.cos(k*c*t)
        hu = -a*c*np.cos(k*x)*np.sin(k*c*t)
        if (A > 0.):
            hw = -0.5*d*a*k*c*np.sin(k*x)*np.sin(k*c*t)
            hpnh = -0.5*a/A*(k*c*d)**2*np.sin(k*x)*np.cos(k*c*t)
            pnh = hpnh/d

        if (A > 0.):
            return h, hu, hw, pnh
        else:
            return h, hu

    def analytical_stand_refl(self, t):
        """
        analytic solution of the non-hydrostatic pressure and the horizontal velocity hu for standing wave test case with reflecting boundaries
        """

        x = self.DGEl.dofcoordinates
        A = self.nhA
        a = 0.01
        l = max(x) - min(x)
        d = self.d
        g = self.g
        k = 2.*np.pi/l
        c = np.sqrt(g*d)
        if (A > 0.):
            c = c/np.sqrt(1.+(k*d)**2/(2.*A))

        h = d-a*np.cos(k*x)*np.cos(k*c*t)
        hu = -a*c*np.sin(k*x)*np.sin(k*c*t)
        if (A > 0.):
            hw = 0.5*d*a*k*c*np.cos(k*x)*np.sin(k*c*t)
            hpnh = 0.5*a/A*(k*c*d)**2*np.cos(k*x)*np.cos(k*c*t)
            pnh = hpnh/d

        if (A > 0.):
            return h, hu, hw, pnh
        else:
            return h, hu

    def analytical_solit(self, t, a, x0):
        """
        analytic solution of the non-hydrostatic pressure and the horizontal velocity hu for solitary wave test case with periodic boundaries
        """

        x = self.DGEl.dofcoordinates
        A = self.nhA
        xlen = max(x) - min(x)
        g = self.g
        d = self.d
        K = np.sqrt(3.*a/(4.*d*d*(d+a)))
        c = np.sqrt(g*(d+a))

        nperiod = int(t*c/xlen)+2
        ssh = np.zeros(len(x))
        hw = np.zeros(len(x))
        hpnh = np.zeros(len(x))

        if (A > 0):
            # loop to make the non-periodic solution somehow periodic
            for i in range(nperiod):
                xhelp = x - x0 - c*t + float(i)*xlen
                ssh_help = a/((np.cosh(K*xhelp))**2)
                T_help = np.tanh(K*xhelp)
                h_help = d + ssh_help
                terms_help = (2.*(T_help**2)*(d/h_help)-ssh_help/a)
                hpnh_help = ((d*c*K)**2)*ssh_help/A*terms_help

                ssh = ssh + ssh_help
                hw = hw + d*c*K*ssh_help*T_help
                hpnh = hpnh + hpnh_help

            h = d + ssh
            hu = c*ssh
            pnh = hpnh/h

        if (A == 0.):
            c = np.sqrt(g*d)
            numf = 1000
            xf = np.zeros(numf)
            for i in range(numf):
                xf[i] = xlen/float(numf-1)*float(i)
            xhelp2 = np.zeros((nperiod, len(xf)))
            for i in range(nperiod):
                xhelp2[i] = xf-c*0.-x0+float(i)*xlen

            fact = float(int(t/30.))
            for i in range(nperiod):
                sshf = a/((np.cosh(K*xhelp2[0]))**2)
                etaf = xhelp2[i]-fact*xlen + x0 + \
                    (3.*np.sqrt(g*(d+sshf))-2.*np.sqrt(g*d))*t
                ssh = ssh + self.func_interpolate(etaf, x, sshf)

            h = d + ssh
            hu = h*2.*(np.sqrt(g*(d+ssh)) - c)

        if (self.nhA > 0.):
            return np.stack((h, hu, hw, pnh), axis=1)
        if (self.nhA == 0.):
            return np.stack((h, hu), axis=1)

    def initial_simplebeach(self, t, a, c, btopo, x0):
        """
        analytic solution of the non-hydrostatic pressure and the horizontal velocity hu for solitary wave test case with periodic boundaries, adjusted for usage with bathymetry
        """

        x = self.DGEl.dofcoordinates
        xlen = max(x) - min(x)
        g = self.g
        d = self.d

        K = np.sqrt(3.*a/(4.*d*d*(d+a)))
        #c = np.sqrt(g*(d))
        #x0 = xlen/2.

        nperiod = int(t*c/xlen)+2
        ssh = np.zeros(len(x))
        hw = np.zeros(len(x))
        pnh = np.zeros(len(x))

        # loop to make the non-periodic solution somehow periodic
        for i in range(nperiod):
            xhelp = x - x0 - c*t + float(i)*xlen
            ssh_help = a/((np.cosh(K*xhelp))**2)
            h_help = d + ssh_help
            if (self.nhA > 0.):
                T_help = np.tanh(K*xhelp)
                terms_help = (2.*(T_help**2)*(d/h_help)-ssh_help/a)
                pnh_help = ((d*c*K)**2)*ssh_help / \
                    self.nhA*terms_help/(d+ssh_help)
                hw = hw + d*c*K*ssh_help*T_help
                pnh = pnh + pnh_help
            ssh = ssh + ssh_help

        h = d + ssh - btopo
        hu = c*ssh

        if (self.nhA > 0.):
            return h, hu, hw, pnh
        if (self.nhA == 0.):
            return h, hu

    def initial_compositebeach(self, t, swnl, a, rlen, xshift):

        # definition of scalars
        g = self.g
        d = self.d
        K = np.sqrt(3.*a/(4.*d*d*(d+a)))
        if (self.nhA > 0.):
            c = np.sqrt(g*(d+a))
        elif (self.nhA == 0.):
            c = np.sqrt(g*d)

        # bathymetry
        x = self.DGEl.dofcoordinates
        xmax = max(x)
        x0 = xmax - rlen - 4.36 - 2.93 - 0.9 - xshift

        m1 = 1./53.
        m2 = 1./150.
        m3 = 1./13.
        x3 = xmax - 0.9
        x2 = x3 - 2.93
        x1 = x2 - 4.36
        # periodic: bv 2x aendern und hu-Vorzeichen
        #x0 = 2.
        # wall: bv 2x aendern und hu-Vorzeichen
        #x0 = xmax-2.

        b = np.zeros(self.DGEl.doflength)
        for i in range(self.DGEl.doflength):
            # real bathymetry
            coo = x[i]
            if ((x1 < coo) & (coo <= x2)):
                b[i] = m1*(coo-x1)
            elif ((x2 < coo) & (coo <= x3)):
                b[i] = m1*(x2-x1) + m2*(coo-x2)
            elif ((x3 < coo) & (coo <= xmax)):
                b[i] = m1*(x2-x1) + m2*(x3-x2) + m3*(coo-x3)
        # for testing: which refinement do I need to keep the amplitude of the solitary wave also in the shallow region on the right wall? There is a higher non-linerity.
            #b[i] = m1*(x2-x1) + m2*(x3-x2) + m3*(max-x3)

        # ssh
        ssh = a/((np.cosh(K*(x-c*t-x0)))**2)
        h = d - b + ssh
        if (self.nhA == 0.):
            if ((swnl == 1) | (swnl == 2)):
                hu = h*2.*(np.sqrt(g*(d+ssh)) - c)
            if (swnl == 0):
                cb = np.sqrt(g*(d-b))
                hu = cb*ssh
        elif (self.nhA > 0.):
            hu = c*ssh

        # if (self.nhA>0.):
            #hpnh = np.zeros(self.DGEl.doflength)
            #T = np.tanh(K*(x-c*t-x0))
            #hw = d*c*K*ssh*T
            # if (a!=0.):
            #hpnh = ((d*K*c)**2)*ssh/self.nhA*(2.*(T**2)*(d/(h))-ssh/a)
            #pnh = hpnh/h

        pnh = np.zeros(self.DGEl.doflength)
        hw = np.zeros(self.DGEl.doflength)

        if (self.nhA > 0.):
            return h, hu, hw, pnh, b
        if (self.nhA == 0.):
            return h, hu, b
