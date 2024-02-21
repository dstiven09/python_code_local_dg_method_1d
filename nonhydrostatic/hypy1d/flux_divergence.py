"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

flux divergences as RHS for time stepping in semi-discetisations of hyperbolic
PDEs
"""

import numpy as np


class FluxDiv1DFV(object):
  """
  finite volume flux divergence as RHS for time stepping
  """

  def __init__(self, Grid, Equation, Recovery, BoundaryFlux):

    self.Gr    = Grid
    self.Eq    = Equation
    self.Rec   = Recovery
    self.BFlux = BoundaryFlux


  def __call__(self, Q, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    FDiv = np.zeros(Q.shape)

    # compute edge values
    [Ql, Qr] = self.Rec(Q, t)

    # flux through left boundary
    #FDiv[0] = bFlux(Ql, Qr, 0);
    #FDiv[0] = self.Eq.num_flux(Qr[-1], Ql[0])

    # flux through inner boundaries
    #for i in range(self.Gr.nlength-2):
      #F         = self.Eq.num_flux(Qr[i], Ql[i+1])
      #FDiv[i]   = (FDiv[i] - F) / self.Gr.elementwidth[i]
      #FDiv[i+1] = F

    # inner edge loop for boundary computations
    for inode in range(self.Gr.inodeelements.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      F      = self.Eq.num_flux(Qr[ndelts[0]], Ql[ndelts[1]])
      FDiv[ndelts[0]] = FDiv[ndelts[0]] - F / self.Gr.elementwidth[ndelts[0]]
      FDiv[ndelts[1]] = FDiv[ndelts[1]] + F / self.Gr.elementwidth[ndelts[1]]

    # boundary edges (interface computations)
    for inode in range(self.Gr.bnodes.shape[0]):

      ndelts = self.Gr.bnodeelements[inode]

      if(inode == 0):
        F = self.BFlux(Ql, Qr, 0, self.Eq.num_flux)
        FDiv[ndelts[1]] = FDiv[ndelts[1]] + F / self.Gr.elementwidth[ndelts[1]]
      else:
        F = self.BFlux(Ql, Qr, 1, self.Eq.num_flux)
        FDiv[ndelts[0]] = FDiv[ndelts[0]] - F / self.Gr.elementwidth[ndelts[0]]

    # flux through right boundary
    #FDiv[-1] = FDiv[:,-1] - bFlux(Ql, Qr, 1);
    #FDiv[-1] = (FDiv[-1] - self.Eq.num_flux(Qr[-1], Ql[0])) / self.Gr.elementwidth[-1]

    # add source term
    return FDiv # + Source(Q, Ql, Qr, t);


class FluxDiv1DFVSrc(FluxDiv1DFV):
  """
  finite volume flux divergence with source term as RHS for time stepping
  """

  def __init__(self, Grid, Equation, Recovery, BoundaryFlux, Source):

    super(FluxDiv1DFVSrc, self).__init__(Grid, Equation, Recovery, BoundaryFlux)
    self.src = Source


  def __call__(self, Q, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    FDiv = super(FluxDiv1DFVSrc, self).__call__(Q, t)

    # compute edge values
    [Ql, Qr] = self.Rec(Q, t)

    return FDiv + self.src(Q, Ql, Qr, t)


class FluxDiv1DDG(object):
  """
  DG flux divergence as RHS for time stepping

  Note: This is just the base class with constructor!
  """

  def __init__(self, Grid, Equation, Src, DGElmt, Quad, BValue):
    self.Gr    = Grid
    self.Eq    = Equation
    self.Src   = Src
    self.DGEl  = DGElmt
    self.Quad  = Quad
    self.bv    = BValue

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

      self.flux[eltdofs] = self.elmtflux(Q[eltdofs], t, self.DGEl.ddx[ielmt])
      self.flux[eltdofs] = self.flux[eltdofs] + \
        self.Src.elmtsource(ielmt, Q[eltdofs], t)

    # inner edge loop for interface computations
    for inode in range(self.Gr.inodes.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      rhsl, rhsr = self.edgeflux(Q[eldofs[-1]], Q[erdofs[0]], t)

      self.flux[eldofs] = self.flux[eldofs] + rhsl/self.DGEl.J[ndelts[0]]
      self.flux[erdofs] = self.flux[erdofs] - rhsr/self.DGEl.J[ndelts[1]]

    # boundary edges (interface computations)
    for inode in range(self.Gr.bnodes.shape[0]):

      ndelts = self.Gr.bnodeelements[inode]
      eidofs = self.DGEl.elementdofs[ndelts[1-self.bv[inode].side],:]

      if (self.bv[inode].side == 0):
        rhsl, rhsr = self.edgeflux(self.bv[inode](Q, t), Q[eidofs[0]], t)

        self.flux[eidofs] = self.flux[eidofs] - rhsr/self.DGEl.J[ndelts[1]]

      else:
        rhsl, rhsr = self.edgeflux(Q[eidofs[-1]], self.bv[inode](Q, t), t)

        self.flux[eidofs] = self.flux[eidofs] + rhsl/self.DGEl.J[ndelts[0]]

    return self.flux


  def elmtflux(self, Qelmt, t, ddxelt):
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
      FDiv = self.Eq.DFlux(qi[iquad], dqi[iquad])

      for idof in range(self.DGEl.edofs):
        rhs[idof,:] = rhs[idof,:] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*FDiv

    return rhs


  def edgeflux(self, ql, qr, t):
    """
    compute flux divergence at interface with left and right states ql and qr
    """

    rhsl = np.zeros((self.DGEl.edofs, self.Eq.unknowns))
    rhsr = np.zeros((self.DGEl.edofs, self.Eq.unknowns))

    Fleft = self.Eq.Flux(ql)
    Frght = self.Eq.Flux(qr)
    Fstar = self.Eq.num_flux(ql, qr)

    for idof in range(self.DGEl.edofs):
      rhsl[idof,:] = self.Quad.gMinvpsi[idof, -1]*(Fleft - Fstar)
      rhsr[idof,:] = self.Quad.gMinvpsi[idof,  0]*(Frght - Fstar)

    return rhsl, rhsr


class FluxDivergence1DDGWeak(FluxDiv1DDG):
  """
  DG flux divergence (weak formulation) as RHS for time stepping

  Note: This object should be independent of the system of equations to solve!
  """

  def __call__(self, Q, Qold, t):
    """
    evaluate discrete flux divergence from state Q at time t
    """

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]

      f = np.sin(np.pi*(self.DGEl.dofcoordinates[eltdofs]+1.))*(np.pi**2)

      self.flux[eltdofs] = self.elmtflux(Q[eltdofs], t, self.DGEl.J[ielmt])

      self.flux[eltdofs] = self.flux[eltdofs] + self.Src.elmtsource(ielmt, Q[eltdofs], Qold[eltdofs], t)


    self.flux = self.edge_computations(self.flux, Q, t)

    return self.flux

  def edge_computations(self, start, Q, t):
    """
    ...
    """

    self.flux = start

    # inner edge loop for boundary computations
    for inode in range(self.Gr.inodes.shape[0]):

      ndelts = self.Gr.inodeelements[inode]
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      erdofs = self.DGEl.elementdofs[ndelts[1],:]

      rhs = self.edgeflux(Q[eldofs[-1]], Q[erdofs[0]], t)

      self.flux[eldofs] = self.flux[eldofs] + np.flipud(rhs)/self.DGEl.J[ndelts[0]]
      self.flux[erdofs] = self.flux[erdofs] - rhs/self.DGEl.J[ndelts[1]]

    # boundary edges (interface computations)
    for inode in range(self.Gr.bnodes.shape[0]):

      ndelts = self.Gr.bnodeelements[inode]
      eidofs = self.DGEl.elementdofs[ndelts[1-self.bv[inode].side],:]

      if (self.bv[inode].side == 0):
        rhs = self.edgeflux(self.bv[inode](Q, t), Q[eidofs[0]], t)

        self.flux[eidofs] = self.flux[eidofs] - rhs/self.DGEl.J[ndelts[1]]

      else:
        rhs = self.edgeflux(Q[eidofs[-1]], self.bv[inode](Q, t), t)

        self.flux[eidofs] = self.flux[eidofs] + np.flipud(rhs)/self.DGEl.J[ndelts[0]]

    return self.flux


  def elmtflux(self, Qelmt, t, J):
    """
    compute inner element flux divergence for one element
    """

    rhs = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    qi   = np.dot(self.Quad.psi.T, Qelmt)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # compute flux divergence at quadrature points
      FDiv = self.Eq.Flux(qi[iquad])

      #print FDiv.shape, rhs.shape

      for idof in range(self.DGEl.edofs):
        rhs[idof,:] = rhs[idof,:] + \
          self.Quad.w[iquad]*self.Quad.eMinvdpsidxi[idof, iquad]/J*FDiv

    return rhs

  def analytical_solit(self, t):
    """
    analytic solution of the non-hydrostatic pressure and the horizontal velocity hu for solitary wave test case with periodic boundaries
    """

    x = self.DGEl.dofcoordinates
    A = 1.5
    xlen = max(x) - min(x)
    g = 9.80616
    d = 10.
    a = 2.
    K = np.sqrt(3.*a/(4.*d*d*(d+a)))
    c = np.sqrt(g*(d+a))
    x0 = xlen/2.

    nperiod = int(t*c/xlen)+2
    ssh = np.zeros(len(x))
    hw  = np.zeros(len(x))
    hpnh= np.zeros(len(x))

    for i in range(nperiod):
      xhelp = x - x0 -c*t + float(i)*xlen
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

    return h, hu

  def elmtfact(self, t, M):
    """
    compute discretized factor for one element
    """

    Mnew = np.zeros(M.shape)

    # compute flux divergence at quadrature points
    for iquad in range(self.Quad.quadpts):
      Mnew[:,iquad] = M[:,iquad] * self.Quad.w[iquad]

    Mnew = np.dot(Mnew,self.Quad.psi.T)

    return Mnew

  def edgeflux(self, ql, qr, t):
    """
    compute flux divergence at interface with left and right states ql and qr
    """

    rhs = np.zeros((self.DGEl.edofs, self.Eq.unknowns))

    Fstar = self.Eq.num_flux(ql, qr)

    for idof in range(self.DGEl.edofs):
      rhs[idof,:] = - self.Quad.gMinvpsi[idof, 0]*Fstar

    return rhs
