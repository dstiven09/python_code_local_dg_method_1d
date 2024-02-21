"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

flux divergences as RHS for time stepping in semi-discetisations of hyperbolic
PDEs
"""

import numpy as np


class FluxDiv1DDG(object):
  """
  DG flux divergence as RHS for time stepping

  Note: This is just the base class with constructor!
  """

  def __init__(self, Grid, Equation, Src, DGElmt, Quad):
    self.Gr    = Grid
    self.Eq    = Equation
    self.Src   = Src
    self.DGEl  = DGElmt
    self.Quad  = Quad

    self.flux  = np.zeros((self.DGEl.doflength, self.Eq.unknowns))


class FluxDiv1DDGStrong(FluxDiv1DDG):
  """
  DG flux divergence (strong formulation) as RHS for time stepping

  Note: This object should be independent of the system of equations to solve!
  """

  def __call__(self, Q, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]

      self.flux[eltdofs] = self.elmtflux(Q[eltdofs], t, self.DGEl.ddx[ielmt],\
                                            self.DGEl.ddxx[ielmt])
      self.flux[eltdofs] = self.flux[eltdofs] + \
        self.Src.elmtsource(ielmt, Q[eltdofs], t)

    # inner edge loop for boundary computations
    for inode in range(self.Gr.inodeelements.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      # left and right derivatives for laplacian in flux
      dQ              = np.zeros(Q.shape)
      ddxql           = np.dot(self.DGEl.ddx[ndelts[0]], self.Quad.psi)
      qil             = np.dot(self.Quad.psi.T, Q[eldofs])
      dqil  = np.dot(ddxql.T, qil)
      dQ[eldofs[-1]]  = dqil[len(dqil)-1]

      ddxqr           = np.dot(self.DGEl.ddx[ndelts[1]], self.Quad.psi)
      qir             = np.dot(self.Quad.psi.T, Q[erdofs])
      dqir            = np.dot(ddxqr.T, qir)
      dQ[erdofs[0]]   = dqir[1]

      # residuals
      rhsl, rhsr = self.edgeflux(Q[eldofs[-1]],dQ[eldofs[-1]],\
                                 Q[erdofs[0]], dQ[erdofs[0]], t)

      self.flux[eldofs] = self.flux[eldofs] + rhsl/self.DGEl.J[ndelts[0]]
      self.flux[erdofs] = self.flux[erdofs] - rhsr/self.DGEl.J[ndelts[1]]

    # boundary edges (for now only extrapolation b.c.)
    for inode in range(self.Gr.lnodeelements.shape[0]):

      ndelts = self.Gr.lnodeelements[inode]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      # right derivatives for laplacian in flux
      dQ              = np.zeros(Q.shape)
      ddxqr           = np.dot(self.DGEl.ddx[ndelts[1]], self.Quad.psi)
      qir             = np.dot(self.Quad.psi.T, Q[erdofs])
      dqir            = np.dot(ddxqr.T, qir)
      dQ[erdofs[0]]   = dqir[1]

      # residuals
      rhsl, rhsr = self.edgeflux(Q[erdofs[0]], dQ[erdofs[0]],\
                                 Q[erdofs[0]], dQ[erdofs[0]], t)

      self.flux[erdofs] = self.flux[erdofs] - rhsr/self.DGEl.J[ndelts[1]]

    for inode in range(self.Gr.rnodeelements.shape[0]):

      ndelts = self.Gr.rnodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]

      # left derivatives for laplacian in flux
      dQ              = np.zeros(Q.shape)
      ddxql           = np.dot(self.DGEl.ddx[ndelts[0]], self.Quad.psi)
      qil             = np.dot(self.Quad.psi.T, Q[eldofs])
      dqil            = np.dot(ddxql.T, qil)
      dQ[eldofs[-1]]  = dqil[len(dqil)-1]

      # residuals
      rhsl, rhsr = self.edgeflux(Q[eldofs[-1]], dQ[eldofs[-1]],\
                                 Q[eldofs[-1]], dQ[eldofs[-1]], t)

      self.flux[eldofs] = self.flux[eldofs] - rhsl/self.DGEl.J[ndelts[0]]

    return self.flux


  def elmtflux(self, Qelmt, t, ddxelt, ddxxelt):
    """
    compute inner element flux divergence for one element
    """

    rhs = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    ddxq = np.dot(ddxelt, self.Quad.psi)
    qi   = np.dot(self.Quad.psi.T, Qelmt[:,0])
    dqi  = np.dot(ddxq.T, Qelmt[:,0])
    ddxxq = np.dot(ddxxelt, self.Quad.psi)
    ddqi  = np.dot(ddxxq.T, Qelmt[:,0])

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # compute flux divergence at quadrature points
      FDiv = self.Eq.DFlux(qi[iquad], dqi[iquad], ddqi[iquad])

      for idof in range(self.DGEl.edofs):
        rhs[idof] = rhs[idof] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*FDiv

    return rhs


  def edgeflux(self, ql, dql, qr, dqr, t):
    """
    compute flux divergence at interface with left and right states ql and qr
    """

    rhsl = np.zeros((self.DGEl.edofs, self.Eq.unknowns))
    rhsr = np.zeros((self.DGEl.edofs, self.Eq.unknowns))

    Fleft = self.Eq.Flux(ql, dql)
    Frght = self.Eq.Flux(qr, dqr)
    Fstar = self.Eq.num_flux(ql,dql, qr, dqr)

    for idof in range(self.DGEl.edofs):
      rhsl[idof,:] = self.Quad.gMinvpsi[idof, -1]*(Fleft - Fstar)
      rhsr[idof,:] = self.Quad.gMinvpsi[idof,  0]*(Frght - Fstar)

    return rhsl, rhsr