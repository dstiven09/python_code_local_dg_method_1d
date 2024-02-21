"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

functionality specific to the advection diffusion equation in "pseudo 1D"
"""

import numpy as np


class EqADE:
  """
  shallow water equations object
  """

  def __init__(self, g=1.0, wettol=1.0e-8):
    self.g        = g
    self.unknowns = 2
    self.wettol   = wettol


  def EVals(self, qi):
    """
    compute Eigen values from state vector qi = (h, 0)
    """

    ci    = np.zeros(len(qi))
    ci[:] = 1.0

    return np.array([ci, ci])


  def LEVec(self, qi):
    """
    compute left Eigen vectors from state vector qi = (h, 0)
    """

    ci    = np.zeros(len(qi))
    ci[:] = 1.0

    return np.array([[ ci, -1],
                     [ ci,  1]])


  def REVec(self, qi):
    """
    compute right Eigen vectors from state vector qi = (h, 0)
    """

    ci    = np.zeros(len(qi))
    ci[:] = 1.0

    return np.array([[1 , 1 ],
                     [ci, ci]])


  def Flux(self, qi, iswet=1.0):
    """
    compute flux vector from state vector qi = (h, 0)
    """

    

    return np.array([qi[0], 0])


  def FluxGrav(self, qi):
    """
    compute flux vector from state vector qi = (h, 0)
    """

    return np.array([qi[0], 0.0])


  def DFlux(self, qi, dqi, iswet=1.0):
    """
    compute spatial derivative of flux vector (flux divergence) from state
    vector qi = (h, 0), and its derivative dqi
    """
    return np.array([dqi[0], 0.0])


  def DFluxAdv(self, qi, dqi):
    """
    compute spatial derivative of ADVECTIVE part of flux vector (flux divergence) from state
    vector qi = (h, 0), and its derivative dqi
    """
    return np.array([dqi[0], 0.0])


class EqADESource:
  """
  source term for shallow water equations (DG discretization)
  """

  def __init__(self, Equation, DGElmt, Quad):
    self.Eq    = Equation
    self.DGEl  = DGElmt
    self.Quad  = Quad

    self.btopo = np.zeros(DGElmt.doflength)

  def update(self, btopo):
    """
    update/initialize source term
    """
    self.btopo = btopo

  def elmtsource(self, ielmt, Qelmt, t):
    """
    compute source term for one element
    """

    si = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    ddxq = np.dot(self.DGEl.ddx[ielmt], self.Quad.psi)
    qi   = np.dot(self.Quad.psi.T, Qelmt)
    dbi  = np.dot(ddxq.T, self.btopo[self.DGEl.elementdofs[ielmt]])

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # add source term at quadrature points
      Src = np.array([0.0, 0.0])

      for idof in range(self.DGEl.edofs):
        si[idof,:] = si[idof,:] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*Src

    return si

class EqADEFVSource:
  """
  source term for shallow water equations (FV discretization)
  """

  def __init__(self, Equation, Grid, reclin):
    self.Eq  = Equation
    self.gr  = Grid
    self.rl  = reclin

    self.btopo = np.zeros(Grid.elength)

  def update(self, btopo):
    """
    update/initialize source term
    """
    self.btopo = btopo

  def __call__(self, Q, Ql, Qr, t):
    """
    compute source term for one element
    """

    Qtmp = np.copy(Q)
    Qtmp[:,1] = Qtmp[:,0]+self.btopo

    Qtmpl, Qtmpr = self.rl(Qtmp, t)
    bl = Qtmpl[:,1] - Qtmpl[:,0]
    br = Qtmpr[:,1] - Qtmpr[:,0]
    hl = Qtmpl[:,0]
    hr = Qtmpr[:,0]

    src      = np.zeros((Q.shape))

    return src
