"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

functionality specific to elliptic equations, using the Local DG method (LDG)
"""

import numpy as np
import scipy.sparse.linalg as linalg
import scipy.sparse as sp
import collections
from projection_criterion import ProjectionCriterionType


class FactorsElliptic:
    """
    discretize factors for each term in the LDG equation system of the elliptic equation,
    according to the input defined in the class Eqell in equation.py
    """

    def __init__(self, Grid, Source, Equell, DGElmt, Quad, Local):
        self.Gr = Grid
        self.Src = Source
        self.Eqell = Equell
        self.DGEl = DGElmt
        self.Quad = Quad
        self.Local = Local

    def update_local_projection_elements(self, flux_divergence, q_new, Qt0,
                                         dg_element, t, projection_criterion):
        """
        gives local area on which to compute the non-hydrostatic projection
        the threasold criterion is set by the parameter nhcrit (floor(nhcrit)=choice of criterion, decimal places: possibly set specific constant for criterion)
        Only entire elements are considered for local area.

        input for routine local_arrays:
        nhcrit: sets the criterion to determine the local non-hydrostatic region
        floor(nhcrit) = 0: criterion acc. to water depth
        floor(nhcrit) = 1: criterion acc. to water depth
        floor(nhcrit) = 2: criterion acc. to dispersion time (Glimsdal et al. (2013))

        return arrays for non-hyrostatic projection:
        lnh:   original dof index
        lelmt: original elmemt index
        dofsnh: dofs of local area
        """

        # initializations and definitions
        lnh, lelmt = [], []
        crit = np.zeros(flux_divergence.Gr.elength, dtype=bool)
        crit_help = np.zeros(flux_divergence.Gr.elength, dtype=bool)
        c0 = np.sqrt(flux_divergence.Eq.g * flux_divergence.Eq.d) #for diagnosys, might not be yet implemented

        # criterion |(h+b-d)/h|
        if (projection_criterion.type ==
                ProjectionCriterionType.H_PLUS_B_MINUS_D_DIV_H):
            b = np.zeros(dg_element.doflength)
            b[:] = self.Eqell.btopo[:]
            x = (q_new[:, 0] + b[:] - flux_divergence.Eq.d) / q_new[:, 0]
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(x[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |hu|
        if (projection_criterion.type == ProjectionCriterionType.HU):
            if t < 0.25:
                for ielmt in range(0, flux_divergence.Gr.elength):
                    crit[ielmt] = True
            else:
                hu = q_new[:, 1]
                for ielmt in range(flux_divergence.Gr.elength):
                    crit[ielmt] = (any(
                        abs(hu[dg_element.elementdofs[ielmt, j]]) >
                        projection_criterion.threshold
                        for j in range(dg_element.edofs)))

        # criterion |u|
        if (projection_criterion == ProjectionCriterionType.U):
            krit = abs(q_new[:, 1] / q_new[:, 0])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(krit[dg_element.elementdofs[ielmt, j]] > (
                    projection_criterion.threshold)
                                   for j in range(dg_element.edofs)))

        # criterion |hw|
        if (projection_criterion.type == ProjectionCriterionType.HW):
            #      if t==0:
            #        for ielmt in range(0,FD.Gr.elength):
            #          crit[ielmt]=True
            #      else:
            hw = q_new[:, 2]
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(hw[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |w|
        if (projection_criterion.type == ProjectionCriterionType.W):
            #      if t==0:
            #        for ielmt in range(0,FD.Gr.elength):
            #          crit[ielmt]=True
            #      else:
            w = q_new[:, 2] / q_new[:, 0]
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(w[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |pnh|
        if (projection_criterion.type == ProjectionCriterionType.PNH):
            #      if t==0:
            #        for ielmt in range(0,FD.Gr.elength):
            #          crit[ielmt]=True
            #      else:
            p = q_new[:, 3]
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(p[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |pnh_x|
        if (projection_criterion.type == ProjectionCriterionType.PNH_X):
            #      if t==0:
            #        for ielmt in range(0,FD.Gr.elength):
            #          crit[ielmt]=True
            #      else:
            px = np.zeros(dg_element.doflength)
            for ielmt in range(0, flux_divergence.Gr.elength):
                px[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    q_new[dg_element.elementdofs[ielmt], 3])
                crit[ielmt] = (any(
                    abs(px[dg_element.elementdofs[ielmt, j]]) > (
                        projection_criterion.threshold)
                    for j in range(dg_element.edofs)))

        # criterion |hw_x|
        if (projection_criterion.type == ProjectionCriterionType.HW_X):
            #      if t==0:
            #        for ielmt in range(0,FD.Gr.elength):
            #          crit[ielmt]=True
            #      else:
            hw = q_new[:, 2]
            hwx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hwx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    hw[dg_element.elementdofs[ielmt]])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(hwx[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |hw_x| with protection against choosing single elements
        if (projection_criterion.type == ProjectionCriterionType.HW_X_PROTECTED
            ):
            #        if t<0.25:
            #          for ielmt in range(0,FD.Gr.elength):
            #            crit[ielmt]=True
            #        else:
            hw = q_new[:, 2]
            hwx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hwx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    hw[dg_element.elementdofs[ielmt]])
            for ielmt in range(1, flux_divergence.Gr.elength - 1):
                crit[ielmt] = (any(
                    abs(hwx[dg_element.elementdofs[ielmt + k, j]]) >
                    projection_criterion.threshold for k in range(-1, 2)
                    for j in range(dg_element.edofs)))
            crit[0] = (any(
                abs(hwx[dg_element.elementdofs[0, j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))
            crit[flux_divergence.Gr.elength - 1] = (any(
                abs(hwx[dg_element.elementdofs[flux_divergence.Gr.elength - 1,
                                               j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))

        # criterion |w_x|
        if (projection_criterion.type == ProjectionCriterionType.W_X):
            #        if t<0.25:
            #          for ielmt in range(0,FD.Gr.elength):
            #            crit[ielmt]=True
            #        else:
            w = q_new[:, 2] / q_new[:, 0]
            wx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                wx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    w[dg_element.elementdofs[ielmt]])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(wx[dg_element.elementdofs[ielmt, j]]) > (
                        projection_criterion.threshold)
                    for j in range(dg_element.edofs)))

        # criterion |w_x| with protection against choosing single elements
        if (projection_criterion.type == ProjectionCriterionType.W_X_PROTECTED
            ):
            w = q_new[:, 2] / q_new[:, 0]
            wx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                wx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    w[dg_element.elementdofs[ielmt]])
            for ielmt in range(1, flux_divergence.Gr.elength - 1):
                crit[ielmt] = (any(
                    abs(wx[dg_element.elementdofs[ielmt + k, j]]) >
                    projection_criterion.threshold for k in range(-1, 2)
                    for j in range(dg_element.edofs)))
            crit[0] = (any(
                abs(wx[dg_element.elementdofs[0, j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))
            crit[flux_divergence.Gr.elength - 1] = (any(
                abs(wx[dg_element.elementdofs[flux_divergence.Gr.elength - 1,
                                              j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))

        # criterion |u_x|
        if (projection_criterion.type == ProjectionCriterionType.U_X):
            u = q_new[:, 1] / q_new[:, 0]
            ux = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                ux[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    u[dg_element.elementdofs[ielmt]])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(ux[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |u_x| with protection against choosing single elements
        if (projection_criterion.type == ProjectionCriterionType.U_X_PROTECTED
            ):
            u = q_new[:, 1] / q_new[:, 0]
            ux = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                ux[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    u[dg_element.elementdofs[ielmt]])
            for ielmt in range(1, flux_divergence.Gr.elength - 1):
                crit[ielmt] = (any(
                    abs(ux[dg_element.elementdofs[ielmt + k, j]]) >
                    projection_criterion.threshold for k in range(-1, 2)
                    for j in range(dg_element.edofs)))
            crit[0] = (any(
                abs(ux[dg_element.elementdofs[0, j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))
            crit[flux_divergence.Gr.elength - 1] = (any(
                abs(ux[dg_element.elementdofs[flux_divergence.Gr.elength - 1,
                                              j]]) >
                projection_criterion.threshold
                for j in range(dg_element.edofs)))

        # criterion |hu_x|
        if (projection_criterion.type == ProjectionCriterionType.HU_X):
            hu = q_new[:, 1]
            hux = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hux[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    hu[dg_element.elementdofs[ielmt]])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(hux[dg_element.elementdofs[ielmt, j]]) >
                    projection_criterion.threshold
                    for j in range(dg_element.edofs)))

        # criterion |hu_x| with protection against choosing single elements
        if (projection_criterion.type == ProjectionCriterionType.HU_X_PROTECTED
            ):
            hu = q_new[:, 1]
            hux = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hux[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    hu[dg_element.elementdofs[ielmt]])
            for ielmt in range(1, flux_divergence.Gr.elength - 1):
                crit[ielmt] = (any(
                    abs(hux[dg_element.elementdofs[ielmt + k, j]]) > (
                        projection_criterion.threshold) for k in range(-1, 2)
                    for j in range(dg_element.edofs)))
            crit[0] = (any(
                abs(hux[dg_element.elementdofs[0, j]]) > (
                    projection_criterion.threshold)
                for j in range(dg_element.edofs)))
            crit[flux_divergence.Gr.elength - 1] = (any(
                abs(hux[dg_element.elementdofs[flux_divergence.Gr.elength - 1,
                                               j]]) >
                (projection_criterion.threshold)
                for j in range(dg_element.edofs)))

        # criterion |h_x|
        if (projection_criterion.type == ProjectionCriterionType.H_X):
            h = q_new[:, 0]
            hx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    h[dg_element.elementdofs[ielmt]])
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = (any(
                    abs(hx[dg_element.elementdofs[ielmt, j]]) > (
                        projection_criterion.threshold)
                    for j in range(dg_element.edofs)))

        # criterion |h_x| with protection against choosing single elements
        if (projection_criterion.type == ProjectionCriterionType.H_X_PROTECTED
            ):
            h = q_new[:, 0]
            hx = np.zeros(dg_element.doflength)
            for ielmt in range(flux_divergence.Gr.elength):
                hx[dg_element.elementdofs[ielmt]] = np.dot(
                    (dg_element.ddx[ielmt]).T,
                    h[dg_element.elementdofs[ielmt]])
            for ielmt in range(1, flux_divergence.Gr.elength - 1):
                crit[ielmt] = (any(
                    abs(hx[dg_element.elementdofs[ielmt + k, j]]) > (
                        projection_criterion.threshold) for k in range(-1, 2)
                    for j in range(dg_element.edofs)))
            crit[0] = (any(
                abs(hx[dg_element.elementdofs[0, j]]) > (
                    projection_criterion.threshold)
                for j in range(dg_element.edofs)))
            crit[flux_divergence.Gr.elength - 1] = (any(
                abs(hx[dg_element.elementdofs[flux_divergence.Gr.elength - 1,
                                              j]]) >
                (projection_criterion.threshold)
                for j in range(dg_element.edofs)))

        if (projection_criterion.type == ProjectionCriterionType.GLOBAL):
            for ielmt in range(flux_divergence.Gr.elength):
                crit[ielmt] = True

        # computation of ouptput arrays
        for i in range(flux_divergence.Gr.elength):
            if (crit[i]):
                # mapping lelmt: nh elements |-> global elements
                lelmt.append(i)
                for j in range(dg_element.edofs):
                    # mapping lnh: nh dofs |-> global dofs
                    lnh.append(dg_element.elementdofs[i, j])
        dofsnh = np.zeros((len(lelmt), dg_element.edofs), dtype=int)
        for ielt in range(len(lelmt)):
            # mapping dofsnh: nh dofs per nh element
            dofsnh[ielt] = ielt * dg_element.edofs + np.arange(
                dg_element.edofs)

        # adjustment for periodic boundary conditions
        if ((len(lnh) > 0) and (len(lnh) < dg_element.doflength)):
            while (lnh[0] == np.mod(lnh[-1] + 1, dg_element.doflength)):
                d = collections.deque(lnh)
                e = collections.deque(lelmt)
                d.rotate(-2)
                e.rotate(-1)
                lnh = list(collections.deque(d))
                lelmt = list(collections.deque(e))

        # convert lists into arrays
        lelmtarr = np.asarray(lelmt)
        lnharr = np.asarray(lnh)

        self.Local.update_local(lnharr, lelmtarr, dofsnh)

    def discr_f1_var2(self, Q):
        """
        discretized factor f1_var2 including gradient
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]

            M = np.zeros((len(eltdofs), len(eltdofs)))
            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            F = self.Eqell.fact_f1_var2(Q[self.Local.dofloc[eltdofs]])

            for idof in range(len(eltdofs)):
                M[:, idof] = - \
                    self.elmtfact(t, self.Quad.dpsidxi)[:, idof] * F[idof]
                for i in range(len(eltdofs)):
                    lcol.append(idof + eltdofs[0] + self.Local.doflenloc)
                    lrow.append(i + eltdofs[0])
                    ldat.append(M[i, idof])

        return lrow, lcol, ldat

    def discr_s1_var1(self, Q, dt):
        """
        discretized factor s1_var1
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            M = np.zeros((len(eltdofs), len(eltdofs)))
            F = self.Eqell.fact_s1_var1(Q[self.Local.dofloc[eltdofs]], bi, dt)

            for idof in range(len(eltdofs)):
                M[:, idof] = self.elmtfact(
                    t, self.Quad.psi)[:, idof] * F[idof] * self.DGEl.J[ielmt]
                for i in range(len(eltdofs)):
                    lcol.append(idof + eltdofs[0])
                    lrow.append(i + eltdofs[0])
                    ldat.append(M[i, idof])

        return lrow, lcol, ldat

    def discr_s1_var2(self, Q):
        """
        discretized factor s1_var2 including gradient
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            M = self.elmtfact_3_s1(ielmt, Q[self.Local.dofloc[eltdofs]],
                                   t) * self.DGEl.J[ielmt]
            for i in range(len(eltdofs)):
                for j in range(len(eltdofs)):
                    lcol.append(j + eltdofs[0] + self.Local.doflenloc)
                    lrow.append(i + eltdofs[0])
                    ldat.append(M[i, j])

        return lrow, lcol, ldat

    def elmtfact_3_s1(self, ielmt, Qelmt, t):
        """
        compute source term as a factor for one element
        """

        Msrc = np.zeros((self.Quad.quadpts, self.DGEl.edofs))
        Mnew = np.zeros((self.DGEl.edofs, self.Quad.quadpts))

        dhi = np.dot((self.DGEl.ddx[ielmt]).T, Qelmt[:, 0])
        dbi = np.dot((self.DGEl.ddx[ielmt]).T,
                     self.Eqell.btopo[self.DGEl.elementdofs[ielmt]])
        bi = self.Eqell.btopo[self.DGEl.elementdofs[ielmt]]
        Fact = self.Eqell.fact_s1_var2(Qelmt, dhi, dbi, bi)
        Mdb = np.dot(self.Quad.psi.T, Fact)

        for iquad in range(self.Quad.quadpts):
            for idof in range(self.DGEl.edofs):
                Msrc[iquad, idof] = Mdb[iquad] * self.Quad.psi.T[iquad, idof]

        for iquad in range(self.Quad.quadpts):
            Mnew[:, iquad] = self.Quad.eMinvpsi[:, iquad] * self.Quad.w[iquad]

        return np.dot(Mnew, Msrc)

    def discr_s1_c(self, Q, dt):
        """
        discretized factor s1_c
        """

        arr = np.zeros(self.Local.doflenloc)
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]
            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            rhs = self.Eqell.fact_s1_c(Q[self.Local.dofloc[eltdofs]], bi, dt)

            arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi),
                                  rhs) * self.DGEl.J[ielmt]

        return arr

    def discr_s1_c_add(self, Q, Qold):
        """
        discretized factor s1_c including gradient
        """

        arr = np.zeros(self.Local.doflenloc)
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            dqiold = np.dot((self.DGEl.ddx[ielmt]).T,
                            Qold[self.Local.dofloc[eltdofs]])
            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            dbi = np.dot((self.DGEl.ddx[ielmt]).T, bi)
            rhs = self.Eqell.fact_s1_c_add(Q[self.Local.dofloc[eltdofs]],
                                           Qold[self.Local.dofloc[eltdofs]],
                                           dqiold, bi, dbi)

            arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi),
                                  rhs) * self.DGEl.J[ielmt]

        return arr

    def discr_f2_var1(self, Q):
        """
        discretized factor f2_var1 including gradient
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]

            M = np.zeros((len(eltdofs), len(eltdofs)))
            F = self.Eqell.fact_f2_var1(Q[self.Local.dofloc[eltdofs]])

            for idof in range(len(eltdofs)):
                M[:, idof] = - \
                    self.elmtfact(t, self.Quad.dpsidxi)[:, idof] * F[idof]
                for i in range(len(eltdofs)):
                    lcol.append(idof + eltdofs[0])
                    lrow.append(i + eltdofs[0] + self.Local.doflenloc)
                    ldat.append(M[i, idof])

        return lrow, lcol, ldat

    def discr_s2_var1(self, Q, dt):
        """
        discretized factor s2_var1
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            M = self.elmtfact_3_s2(ielmt, Q[self.Local.dofloc[eltdofs]], dt,
                                   t) * self.DGEl.J[ielmt]
            for i in range(len(eltdofs)):
                for j in range(len(eltdofs)):
                    lcol.append(j + eltdofs[0])
                    lrow.append(i + eltdofs[0] + self.Local.doflenloc)
                    ldat.append(M[i, j])

        return lrow, lcol, ldat

    def elmtfact_3_s2(self, ielmt, Qelmt, dt, t):
        """
        compute source term as a matrix factor in one element
        """

        Msrc = np.zeros((self.Quad.quadpts, self.DGEl.edofs))
        Mnew = np.zeros((self.DGEl.edofs, self.Quad.quadpts))

        dhi = np.dot((self.DGEl.ddx[ielmt]).T, Qelmt[:, 0])
        bi = self.Eqell.btopo[self.DGEl.elementdofs[ielmt]]
        dbi = np.dot((self.DGEl.ddx[ielmt]).T, bi)
        Fact = self.Eqell.fact_s2_var1(Qelmt, dhi, bi, dbi, dt)
        Mdb = np.dot(self.Quad.psi.T, Fact)

        for iquad in range(self.Quad.quadpts):
            for idof in range(self.DGEl.edofs):
                Msrc[iquad, idof] = Mdb[iquad] * self.Quad.psi.T[iquad, idof]

        for iquad in range(self.Quad.quadpts):
            Mnew[:, iquad] = self.Quad.psi[:, iquad] * self.Quad.w[iquad]

        return np.dot(Mnew, Msrc)

    def discr_s2_var2(self, Q, dt):
        """
        discretized factor s2_var2
        """

        lrow, lcol, ldat = [], [], []
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            M = np.zeros((len(eltdofs), len(eltdofs)))
            F = self.Eqell.fact_s2_var2(Q[self.Local.dofloc[eltdofs]], bi, dt)

            for idof in range(len(eltdofs)):
                M[:, idof] = self.elmtfact(
                    t, self.Quad.psi)[:, idof] * F[idof] * self.DGEl.J[ielmt]
                for i in range(len(eltdofs)):
                    lcol.append(idof + eltdofs[0] + self.Local.doflenloc)
                    lrow.append(i + eltdofs[0] + self.Local.doflenloc)
                    ldat.append(M[i, idof])

        return lrow, lcol, ldat

    def discr_s2_c(self, Q):
        """
        discretized factor s2_c
        """

        lrow, lcol, ldat = [], [], []
        arr = np.zeros(self.Local.doflenloc)
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            rhs = self.Eqell.fact_s2_c(Q[self.Local.dofloc[eltdofs]], bi,
                                       ielmt)

            arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi),
                                  rhs) * self.DGEl.J[ielmt]

        return arr

    def discr_s2_c_add(self, Q, Qold, dt):
        """
        discretized factor s1_c including gradient
        """

        arr = np.zeros(self.Local.doflenloc)
        t = 0.

        # element loop
        for ie in range(self.Local.elmtlenloc):
            eltdofs = self.Local.eltdofsloc[ie]
            ielmt = self.Local.elmtloc[ie]

            bi = self.Eqell.btopo[self.Local.dofloc[eltdofs]]
            rhs = self.Eqell.fact_s2_c_add(Q[self.Local.dofloc[eltdofs]],
                                           Qold[self.Local.dofloc[eltdofs]],
                                           bi, dt)

            arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi),
                                  rhs) * self.DGEl.J[ielmt]

        return arr

    def elmtfact(self, t, M):
        """
        compute discretized factor for one element
        """

        Mnew = np.zeros(M.shape)

        # compute flux divergence at quadrature points
        for iquad in range(self.Quad.quadpts):
            Mnew[:, iquad] = M[:, iquad] * self.Quad.w[iquad]

        return np.dot(Mnew, self.Quad.psi.T)


class EllipticSolver:
    """
    solve the elliptic equation
    """

    def Solve_sep(self, RHS, Q):
        """
        for debugging: solve each of both equations seperately
        """

        Q = RHS(Q, 0)

        return Q

    def solve_linalg(self, A, b):
        """
        solve equation system with numpy routine linalg.solve
        """

        # print np.linalg.cond(A.toarray())

        x = np.linalg.solve(A.toarray(), b)
        if (np.allclose(np.dot(A.toarray(), x), b) == True):
            # print('System solved successfully')
            pass
        else:
            print('Attention: System not solved properly!')
            print('Explicitly terminated.')
            quit()
        rank = np.linalg.matrix_rank(A.toarray())
        if (rank < len(b)):
            print('Attention: Matrix has no full rank! ', 'dofs-rank= ',
                  len(b) - rank)

        return x

    def solve_gmres(self, A, b, Q, dofs1, dofs2):
        """
        solve equation system with lgmres algorithm
        """

        # guess for x: take solution of last timestep
        doflen = len(dofs1)
        xtry = np.zeros(2 * doflen)
        xtry[dofs1] = Q[:, 1]
        xtry[dofs2] = Q[:, 3]

        x, i = linalg.lgmres(A, b, xtry, 1e-12)
        if i == 0:
            print('System solved successfully')
        elif i > 0:
            print(
                'convergence to tolerance not achieved, number of iterations: ',
                i)
            f = open(filestr, 'a')
            f.write(str(doflen / 2 + 1) + ", " + str(t) + '\n')
            f.close()
        else:
            print('illegal input or breakdown')

        return x

    def solve_schur(self, A1, b, Q, dofs1, dofs2):
        """
        solve equation system with Schur complement
        """

        # decomposition of matrix A1 into its 4 block matrizes:
        doflen = len(dofs1)
        x = np.zeros(2 * doflen)
        A = (A1.toarray())[0:doflen, 0:doflen]
        B = A1.toarray()[0:doflen, doflen:2 * doflen]
        C = A1.toarray()[doflen:2 * doflen, 0:doflen]
        D = A1.toarray()[doflen:2 * doflen, doflen:2 * doflen]

        CAm1 = np.dot(C, np.linalg.inv(A))
        S2 = D - np.dot(CAm1, B)
        b2 = b[dofs2] - np.dot(CAm1, b[dofs1])

        x[dofs2], i1 = linalg.lgmres(S2, b2, np.zeros(doflen), 1e-12)
        if i1 == 0:
            print('System solved successfully: dofs2')
        elif i1 > 0:
            print(
                'convergence to tolerance not achieved, number of iterations: ',
                i1)
        else:
            print('illegal input or breakdown')

        b1 = b[dofs1] - np.dot(B, x[dofs2])

        x[dofs1], i2 = linalg.lgmres(A, b1, np.zeros(doflen), 1e-12)
        if i2 == 0:
            print('System solved successfully: dofs1')
        elif i2 > 0:
            print(
                'convergence to tolerance not achieved, number of iterations: ',
                i1)
        else:
            print('illegal input or breakdown')

        return x

    def coo_matrix_arrays(self, fact, Q, dt, Bvell):
        """
        construct sparse matrix in format coo_matrix
        """

        # inner components
        lr, lc, ld = fact.discr_s1_var1(Q, dt)
        lr1, lc1, ld1 = fact.discr_f1_var2(Q)
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = fact.discr_s1_var2(Q)
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = fact.discr_f2_var1(Q)
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = fact.discr_s2_var1(Q, dt)
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = fact.discr_s2_var2(Q, dt)
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        # boundary conditions
        lr1, lc1, ld1 = Bvell[0].bc11()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[0].bc12()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[0].bc21()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[0].bc22()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[1].bc11()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[1].bc12()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[1].bc21()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        lr1, lc1, ld1 = Bvell[1].bc22()
        lr = np.concatenate((lr, lr1))
        lc = np.concatenate((lc, lc1))
        ld = np.concatenate((ld, ld1))

        return lr, lc, ld

    def solve_system(self, fact, Q, dt, t, Bvell, Qold):
        """
        construct and solve the linear equation system for the non-hydrostatic pressure (second unknown) and the updated horizontal velocity hu (first unknown)
        """

        # store local arrays
        lnh = fact.Local.dofloc
        lelmt = fact.Local.elmtloc
        dofsnh = fact.Local.eltdofsloc

        # set boundary conditions
        lennh = fact.Local.doflenloc
        lenQ = len(Q[:, 0])
        bc_1 = np.zeros(lennh)
        bc_2 = np.zeros(lennh)
        if (len(Bvell[2]) == 2):  # non-zero Dirichlet
            F = np.zeros(lennh)
            F[0] = Bvell[2][0]
            F[-1] = Bvell[2][1]
            lr, lc, ld = fact.discr_f1_var2(Q)
            B = sp.coo_matrix((ld, (lr, lc)),
                              shape=(lennh * 2, lennh * 2)).toarray()
            bc_1 = -np.dot(B[0:lennh, lennh:lennh * 2], F)
            bc_2[0] = Bvell[2][0]
            bc_2[-1] = -Bvell[2][1]

        # adjustments for 2nd order timestepping
        if (fact.Eqell.nht == 2):
            rhs1_2nd = fact.discr_s1_c_add(Q, Qold)
            rhs2_2nd = fact.discr_s2_c_add(Q, Qold, dt)
        else:
            rhs1_2nd = np.zeros(lennh)
            rhs2_2nd = np.zeros(lennh)

        # set up right-hand side
        b = np.zeros(2 * lennh)
        dofs1 = np.arange(0, lennh, 1)
        dofs2 = np.arange(lennh, lennh * 2, 1)
        b[dofs1] = fact.discr_s1_c(Q, dt) + bc_1 + bc_2 + rhs1_2nd
        b[dofs2] = fact.discr_s2_c(Q) + rhs2_2nd

        # set up and solve linear equation system
        row, col, data = self.coo_matrix_arrays(fact, Q, dt, Bvell)
        A = sp.coo_matrix((data, (row, col)), shape=(lennh * 2, lennh * 2))

        y = self.solve_linalg(A, b)

        # store values of y in x
        x = np.zeros(2 * lenQ)
        dofsQ = np.arange(0, fact.DGEl.doflength, 1)
        dofs = np.arange(0, lennh, 1)
        x[lnh] = y[dofs]
        x[[i + lenQ for i in lnh]] = y[dofs + lennh]

        # update non-zero Dirichlet boundary conditions to solution of zero boundary problem
        if len(Bvell[2]) == 2:
            x[len(dofs) + dofs[0]] = x[len(dofs) + dofs[0]] + Bvell[2][0]
            x[len(dofs) + dofs[-1]] = x[len(dofs) + dofs[-1]] + Bvell[2][1]

        # wet-dry treatment
        mask = Q[:, 0] > fact.Eqell.wettol
        xhelp = np.zeros(2 * len(dofsQ))
        arr = np.arange(lenQ, lenQ * 2, 1)
        xhelp[dofsQ[mask]] = x[dofsQ[mask]]
        xhelp[arr[mask]] = x[arr[mask]]

        return xhelp[dofsQ], xhelp[dofsQ + len(dofsQ)]

    def correct_tracer(self, fact, FD, Qnew, hu, pnh, t, dt, Qold):
        """
        correction step: update h and hw, copy hu
        """

        # initializations
        Qnew2 = np.zeros(Qnew.shape)
        hw = np.zeros(Qnew[:, 2].shape)

        # in case of second order timestepping: h-corrector
        if (fact.Eqell.nht == 2):
            h = np.zeros(fact.DGEl.doflength)
            start = np.zeros((fact.DGEl.doflength, FD.Eq.unknowns))
            Qhelp = np.zeros((fact.DGEl.doflength, FD.Eq.unknowns))
            Qhelp[:, 0] = Qnew[:, 0]
            Qhelp[:, 1] = hu - Qnew[:, 1]

            # element loop:
            for ielmt in range(FD.Gr.elength):
                eltdofs = fact.DGEl.elementdofs[ielmt]
                flxh = fact.Eqell.corr_hflx(Qnew[eltdofs], hu[eltdofs], dt)
                h[eltdofs] = Qnew[eltdofs, 0] + \
                    np.dot(fact.elmtfact(t, fact.Quad.eMinvdpsidxi),
                           flxh) / fact.DGEl.J[ielmt]

            edgeflx = FD.edge_computations(start, Qhelp, t)
            h = h + dt * edgeflx[:, 0]
            Qnew2[:, 0] = h
        if (fact.Eqell.nht <= 1):
            Qnew2[:, 0] = Qnew[:, 0]

        # wet-dry treatment
        mask = Qnew2[:, 0] > fact.Eqell.wettol
        xhelp = np.zeros(fact.DGEl.doflength)
        xhelp[mask] = Qnew2[mask, 0]
        Qnew2[:, 0] = xhelp

        # hw-corrector using corrected h (which is already corrected in case of nht=2)
        for ielmt in range(FD.Gr.elength):
            eltdofs = fact.DGEl.elementdofs[ielmt]
            rhs_hw = fact.Eqell.corr_hw(Qnew[eltdofs], pnh[eltdofs], dt,
                                        Qold[eltdofs])
            hw[eltdofs] = np.dot(fact.elmtfact(t, FD.Quad.eMinvpsi), rhs_hw)

        Qnew2[:, 1] = hu
        Qnew2[:, 2] = hw
        Qnew2[:, 3] = pnh

        return Qnew2
