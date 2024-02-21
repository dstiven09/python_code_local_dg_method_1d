"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013), Nicole Beisiegel (2014)

flux divergences as RHS for time stepping in semi-discetisations of hyperbolic
PDEs
"""

import numpy as np
from .flux_divergence import FluxDivergence1DDGWeak
from .flux_divergence import FluxDiv1DDGStrong


class FluxDiv1DDGStrongSWE(FluxDiv1DDGStrong):
  """
  DG flux divergence (strong formulation ) as RHS for time stepping

  This is a specific implementation for SWE with wetting and drying
  """

  def __init__(self, Grid, Equation, Src, DGElmt, Quad, BValue, wettol=1.0e-8):

    super(FluxDiv1DDGStrongSWE, self).__init__(Grid, Equation, Src, DGElmt, Quad, BValue)
    self.wettol = wettol

  def __call__(self, Q, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]

      Qelt  = Q[eltdofs]
      belt  = self.Src.btopo[eltdofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswet = 0.0
      else:
        iswet = 1.0

      self.flux[eltdofs] = self.elmtflux(Q[eltdofs], t, self.DGEl.ddx[ielmt], iswet)
      self.flux[eltdofs] = self.flux[eltdofs] + iswet*self.Src.elmtsource(ielmt, Q[eltdofs], t)

    # inner edge loop for interface computations
    for inode in range(self.Gr.inodes.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      ql = Q[eldofs[-1]]
      qr = Q[erdofs[ 0]]

      # check if elements are not semidry of flooding type
      Qelt  = Q[eldofs]
      belt  = self.Src.btopo[eldofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetl = 0.0
      else:
        iswetl = 1.0

      Qelt  = Q[erdofs]
      belt  = self.Src.btopo[erdofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetr = 0.0
      else:
        iswetr = 1.0

      rhsl, rhsr = self.edgeflux(Q[eldofs[-1]], Q[erdofs[0]], self.Src.btopo[eldofs[-1]],
                                 self.Src.btopo[erdofs[0]], t, iswetl, iswetr)

      self.flux[eldofs] = self.flux[eldofs] + rhsl/self.DGEl.J[ndelts[0]]
      self.flux[erdofs] = self.flux[erdofs] - rhsr/self.DGEl.J[ndelts[1]]

    # boundary edges (interface computations)
    for inode in range(self.Gr.bnodes.shape[0]):

      ndelts = self.Gr.bnodeelements[inode]
      eidofs = self.DGEl.elementdofs[ndelts[1-self.bv[inode].side],:]

      Qelt  = Q[eidofs]
      belt  = self.Src.btopo[eidofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetb = 0.0
      else:
        iswetb = 1.0

      Qbfull = self.bv[inode](np.concatenate((Q, np.array([self.Src.btopo]).T), axis=1), t)

      if (self.bv[inode].side == 0):
        rhsl, rhsr = self.edgeflux(Qbfull[0:2], Q[eidofs[0]], Qbfull[2],
                                   self.Src.btopo[eidofs[0]], t, iswetb, iswetb)

        self.flux[eidofs] = self.flux[eidofs] - rhsr/self.DGEl.J[ndelts[1]]

      else:
        rhsl, rhsr = self.edgeflux(Q[eidofs[-1]], Qbfull[0:2], self.Src.btopo[eidofs[-1]],
                                   Qbfull[2], t, iswetb, iswetb)

        self.flux[eidofs] = self.flux[eidofs] + rhsl/self.DGEl.J[ndelts[0]]

    return self.flux


  def elmtflux(self, Qelmt, t, ddxelt, iswet):
    """
    compute inner element flux divergence for one element
    """

    rhs = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    ddxq = np.dot(ddxelt, self.Quad.psi)
    qi   = np.dot(self.Quad.psi.T, Qelmt)
    dqi  = np.dot(ddxq.T, Qelmt)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # compute flux divergence at quadrature points
      FDiv = self.Eq.DFlux(qi[iquad], dqi[iquad], iswet)

      for idof in range(self.DGEl.edofs):
        rhs[idof,:] = rhs[idof,:] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*FDiv

    return rhs


  def edgeflux(self, ql, qr, bl, br, t, iswetl, iswetr):
    """
    compute flux divergence at interface with left and right states ql and qr
    """

    rhsl = np.zeros((self.DGEl.edofs, self.Eq.unknowns))
    rhsr = np.zeros((self.DGEl.edofs, self.Eq.unknowns))

    if (ql[0] > self.wettol or qr[0] > self.wettol):

      Fleft = self.Eq.Flux(ql) #iswetl
      Frght = self.Eq.Flux(qr) #iswetr

      if (qr[0] <= self.wettol and ql[0]+bl < br):
        # right side dry, left side wet, wall case
        qr[0] =  ql[0]
        qr[1] = -ql[1]
        Fstar = self.Eq.num_flux(ql, qr)
        Frght = Fstar

      elif (ql[0] <= self.wettol and qr[0]+br < bl):
        # left side dry, right side wet, wall case
        ql[0] =  qr[0]
        ql[1] = -qr[1]
        Fstar = self.Eq.num_flux(ql, qr)
        Fleft = Fstar

      else:
        # fix for discontinuous bathymetry (Audusse et al.,2004, Xing, Zhang & Shu, 2010)
        ql_star = np.zeros(ql.shape)
        qr_star = np.zeros(qr.shape)
        ql_star[0] = max(0.0, ql[0] + bl - max(bl, br))
        ql_star[1] = ql_star[0]*self.Eq.u(ql)
        qr_star[0] = max(0.0, qr[0] + br - max(br, bl))
        qr_star[1] = qr_star[0]*self.Eq.u(qr)

        Fstar = self.Eq.num_flux(ql_star, qr_star)

        Fleft = Fleft + (self.Eq.FluxGrav(ql_star) - self.Eq.FluxGrav(ql))
        Frght = Frght + (self.Eq.FluxGrav(qr_star) - self.Eq.FluxGrav(qr))

      for idof in range(self.DGEl.edofs):
        rhsl[idof,:] = self.Quad.gMinvpsi[idof, -1]*(Fleft - Fstar)
        rhsr[idof,:] = self.Quad.gMinvpsi[idof,  0]*(Frght - Fstar)

    return rhsl, rhsr


class FluxDiv1DDGWeakSWE(FluxDivergence1DDGWeak):
  """
  DG flux divergence (weak formulation) as RHS for time stepping

  This is a specific implementation for SWE with wetting and drying
  """

  def __init__(self, Grid, Equation, Src, DGElmt, Quad, BValue, wettol=1.0e-8):

    super(FluxDiv1DDGWeakSWE, self).__init__(Grid, Equation, Src, DGElmt, Quad, BValue)
    self.wettol = wettol


  def __call__(self, Q, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    f = np.finfo(float)

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]

      Qelt  = Q[eltdofs]
      belt  = self.Src.btopo[eltdofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswet = 0.0
      else:
        iswet = 1.0

      self.flux[eltdofs] = self.elmtflux(Q[eltdofs], t, self.DGEl.J[ielmt], iswet)
      self.flux[eltdofs] = self.flux[eltdofs] + iswet*self.Src.elmtsource(ielmt, Q[eltdofs], t)

    # inner edge loop for interface computations
    for inode in range(self.Gr.inodes.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      ql = Q[eldofs[-1]]
      qr = Q[erdofs[ 0]]

      # check if elements are not semidry of flooding type
      Qelt  = Q[eldofs]
      belt  = self.Src.btopo[eldofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetl = 0.0
      else:
        iswetl = 1.0

      Qelt  = Q[erdofs]
      belt  = self.Src.btopo[erdofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetr = 0.0
      else:
        iswetr = 1.0

      rhsl, rhsr = self.edgeflux(Q[eldofs[-1]], Q[erdofs[0]], self.Src.btopo[eldofs[-1]],
                                 self.Src.btopo[erdofs[0]], t, iswetl, iswetr)

      self.flux[eldofs] = self.flux[eldofs] + rhsl/self.DGEl.J[ndelts[0]]
      self.flux[erdofs] = self.flux[erdofs] - rhsr/self.DGEl.J[ndelts[1]]

    # boundary edges (interface computations)
    for inode in range(self.Gr.bnodes.shape[0]):

      ndelts = self.Gr.bnodeelements[inode]
      eidofs = self.DGEl.elementdofs[ndelts[1-self.bv[inode].side],:]

      Qelt  = Q[eidofs]
      belt  = self.Src.btopo[eidofs]
      imaxh = np.argmax(Qelt[:,0]+belt)
      #if (np.min(Qelt[:,0]) < self.wettol and Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
      if (Qelt[imaxh,0]+belt[imaxh]-np.max(belt) < self.wettol):
        iswetb = 0.0
      else:
        iswetb = 1.0

      Qbfull = self.bv[inode](np.concatenate((Q, np.array([self.Src.btopo]).T), axis=1), t)

      if (self.bv[inode].side == 0):
        rhsl, rhsr = self.edgeflux(Qbfull[0:2], Q[eidofs[0]], Qbfull[2],
                                   self.Src.btopo[eidofs[0]], t, iswetb, iswetb)

        self.flux[eidofs] = self.flux[eidofs] - rhsr/self.DGEl.J[ndelts[1]]

      else:
        rhsl, rhsr = self.edgeflux(Q[eidofs[-1]], Qbfull[0:2], self.Src.btopo[eidofs[-1]],
                                   Qbfull[2], t, iswetb, iswetb)

        self.flux[eidofs] = self.flux[eidofs] + rhsl/self.DGEl.J[ndelts[0]]

    return self.flux


  def elmtflux(self, Qelmt, t, J, iswet):
    """
    compute inner element flux divergence for one element
    """

    rhs = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    qi = np.dot(self.Quad.psi.T, Qelmt)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # compute flux divergence at quadrature points
      FDiv = self.Eq.Flux(qi[iquad], iswet)

      for idof in range(self.DGEl.edofs):
        rhs[idof,:] = rhs[idof,:] + \
          self.Quad.w[iquad]*self.Quad.eMinvdpsidxi[idof, iquad]/J*FDiv

    return rhs


  def edgeflux(self, ql, qr, bl, br, t, iswetl, iswetr):
    """
    compute flux divergence at interface with left and right states ql and qr
    """

    rhsl = np.zeros((self.DGEl.edofs, self.Eq.unknowns))
    rhsr = np.zeros((self.DGEl.edofs, self.Eq.unknowns))

    if (ql[0] > self.wettol or qr[0] > self.wettol):

      if (qr[0] <= self.wettol and ql[0]+bl < br):
        # right side dry, left side wet, wall case
        qr[0] =  ql[0]
        qr[1] = -ql[1]
        Fstar = self.Eq.num_flux(ql, qr)
        Fleft = 0.0
        Frght = Fstar

      elif (ql[0] <= self.wettol and qr[0]+br < bl):
        # left side dry, right side wet, wall case
        ql[0] =  qr[0]
        ql[1] = -qr[1]
        Fstar = self.Eq.num_flux(ql, qr)
        Fleft = Fstar
        Frght = 0.0

      else:
        # fix for discontinuous bathymetry (Audusse et al.,2004, Xing, Zhang & Shu, 2010)
        ql_star = np.zeros(ql.shape)
        qr_star = np.zeros(qr.shape)
        ql_star[0] = max(0.0, ql[0] + bl - max(bl, br))
        ql_star[1] = ql_star[0]*self.Eq.u(ql)
        qr_star[0] = max(0.0, qr[0] + br - max(br, bl))
        qr_star[1] = qr_star[0]*self.Eq.u(qr)

        Fstar = self.Eq.num_flux(ql_star, qr_star)

        Fleft = (1.0-iswetl)*self.Eq.FluxGrav(ql) + (self.Eq.FluxGrav(ql_star) - self.Eq.FluxGrav(ql))
        Frght = (1.0-iswetr)*self.Eq.FluxGrav(qr) + (self.Eq.FluxGrav(qr_star) - self.Eq.FluxGrav(qr))

      for idof in range(self.DGEl.edofs):
        rhsl[idof,:] = - self.Quad.gMinvpsi[idof, -1]*(Fstar - Fleft)
        rhsr[idof,:] = - self.Quad.gMinvpsi[idof,  0]*(Fstar - Frght)

    return rhsl, rhsr
