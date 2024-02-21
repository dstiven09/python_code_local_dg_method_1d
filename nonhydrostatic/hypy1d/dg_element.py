"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

nodal DG infrastructure for 1D computations
"""

import numpy as np
import hypy1d.interpolation as interpolation
import hypy1d.quadrature as quadrature


class DGReferenceElement1D:
  """
  nodal DG Reference element on the interval [-1, 1]
  """

  def __init__(self, N):
    self.N    = N
    if (N==0):
      self.r = 0.0
    else:
      self.r = quadrature.JacobiGL(0, 0, N)
    self.V    = interpolation.Vandermonde1D(N, self.r)
    self.Vinv = np.linalg.inv(self.V)
    self.Minv = np.dot(self.V, self.V.T)
    self.M    = np.dot(self.Vinv.T, self.Vinv)
    self.Dr   = interpolation.Dmatrix1D(N, self.r, self.V)

class DGElement1D:
  """
  nodal DG discretisation derived from a given grid and DG reference element
  """

  def __init__(self, Grid, RefElmt):

    self.edofs          = RefElmt.N+1
    self.J              = Grid.elementwidth/2.0
    self.doflength      = self.edofs*Grid.elength
    self.elementdofs    = np.zeros((Grid.elength, self.edofs), dtype=int)
    self.dofcoordinates = np.zeros(self.doflength)
    self.ddx            = np.zeros((Grid.elength, self.edofs, self.edofs))
    self.ddxx           = np.zeros((Grid.elength, self.edofs, self.edofs))

    for ielt in range(Grid.elength):
      self.elementdofs[ielt] = ielt*self.edofs+np.arange(self.edofs)
      self.dofcoordinates[self.elementdofs[ielt]] = \
        (RefElmt.r+1.0)*self.J[ielt]+Grid.nodecoordinates[ielt]
      self.ddx[ielt] = RefElmt.Dr/self.J[ielt]
      self.ddxx[ielt] = np.dot(RefElmt.Dr,RefElmt.Dr)/self.J[ielt]**2

def DGProlong(QQc, RefElmt, DGElmtc, DGElmtf):
  """
  prolongate DG function from coarse function space to fine one, where the first
  is "enthalten" by the latter
  """
  f   = np.finfo(float)
  QQf = np.zeros(DGElmtf.doflength)

  for ielmt in range(DGElmtf.elementdofs.shape[0]):
    findc0 = np.min(DGElmtf.dofcoordinates[DGElmtf.elementdofs[ielmt,:]]) >= \
             np.min(DGElmtc.dofcoordinates[DGElmtc.elementdofs] - f.resolution, axis=1)
    findc1 = np.max(DGElmtf.dofcoordinates[DGElmtf.elementdofs[ielmt,:]]) <= \
             np.max(DGElmtc.dofcoordinates[DGElmtc.elementdofs] + f.resolution, axis=1)
    findc  = findc0 & findc1

    dofcooc = DGElmtc.dofcoordinates[DGElmtc.elementdofs[findc]][0]
    dofcoof = DGElmtf.dofcoordinates[DGElmtf.elementdofs[ielmt]]

    dofxrel = (dofcoof-np.min(dofcooc))/(np.max(dofcooc)-np.min(dofcooc))*2.0-1.0
    psi     = np.dot(interpolation.Vandermonde1D(RefElmt.N, dofxrel), RefElmt.Vinv)
    QQf[DGElmtf.elementdofs[ielmt]] = np.dot(psi, QQc[DGElmtc.elementdofs[findc]].T)

  return QQf


class DGQuadrature1D:
  """
  Gauss Legendre quadrature positions, weights and basis function evaluations
  for a DG reference element
  """

  def __init__(self, RefElmt, N):

    f = np.finfo(float)

    self.quadpts   = N+1
    self.x, self.w = quadrature.JacobiGQ(0, 0, N)
    self.x[abs(self.x) < f.resolution] = 0.0

    # compute quadrature point values of nodal (Lagrange) basis function
    self.psi          = np.dot(interpolation.Vandermonde1D(RefElmt.N, self.x), RefElmt.Vinv).T
    self.gMinvpsi     = RefElmt.Minv # Note: This should really be Minv*psi at the boundaries, but should be correct for GL interpolation points...
    self.eMinvpsi     = np.dot(RefElmt.Minv, self.psi)
    self.eMinvdpsidxi = np.dot(np.dot(RefElmt.Minv, RefElmt.Dr), self.psi)
    self.dpsidxi      = np.dot(RefElmt.Dr, self.psi)

def L2norm(Q, DGElmt, Quad):
  """
  compute L2 norm for a DG discretization
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]

    Qqsq   = np.dot(Quad.psi.T, Q[eltdofs])**2
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return np.sqrt(IQsq)

def L2SKP(Q, f2, Gr, DGElmt, Quad, iana):
  """
  compute L2 norm for a DG discretization
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    xq = (Quad.x+1.0)/2.0 * (eltcoo[1]-eltcoo[0]) + eltcoo[0]

    if (iana==0):
      Qqsq   = np.dot(Quad.psi.T, Q[eltdofs])*np.dot(Quad.psi.T, f2[eltdofs])
    if (iana==1):
      Qqsq   = np.dot(Quad.psi.T, Q[eltdofs])*f2(xq)
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return np.sqrt(IQsq)

def L2projection(Gr, DGElmt, Quad, f, iana):
  """
  compute L2 projection of a given function f according to a DG discretization
  """
  fL2 = np.zeros(DGElmt.doflength)

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    xq = (Quad.x+1.0)/2.0 * (eltcoo[1]-eltcoo[0]) + eltcoo[0]

    if (iana==1):
      fL2[eltdofs] = np.dot(Quad.eMinvpsi, Quad.w*f(xq))
    if (iana==0):
      fL2[eltdofs] = np.dot(Quad.eMinvpsi, Quad.w*np.dot(Quad.psi.T, f[eltdofs]))

  # test of routine
  print(L2SKP(fL2, fL2, Gr, DGElmt, Quad, 0) - L2SKP(fL2, f, Gr, DGElmt, Quad, iana))
  print('L2-error: ', L2error(fL2, Gr, DGElmt, Quad, f, iana))

  return fL2

def L2error(Q, Gr, DGElmt, Quad, f, iconvana):
  """
  compute L_2 error of a DG discretization with respect to a given function f
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    xq = (Quad.x+1.0)/2.0 * (eltcoo[1]-eltcoo[0]) + eltcoo[0]

    if (iconvana==1):
      Qqsq   = (np.dot(Quad.psi.T, Q[eltdofs]) - f(xq))**2
    else:
      Qqsq   = (np.dot(Quad.psi.T, Q[eltdofs]) - np.dot(Quad.psi.T, f[eltdofs]))**2
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return np.sqrt(IQsq)

def Linferror(Q, Gr, DGElmt, RefElmt, intpts, f, iconvana):
  """
  compute L_inf error of a DG discretization with respect to a given function f
  """
  errMax = 0.0
  intxre  = np.linspace(-1.0, 1.0, intpts)
  intpsi  = np.dot(interpolation.Vandermonde1D(RefElmt.N, intxre), RefElmt.Vinv)

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    xint    = (intxre+1.0)/2.0 * (eltcoo[1]-eltcoo[0]) + eltcoo[0]

    if (iconvana==1):
      errloc = np.max(abs(np.dot(intpsi, Q[eltdofs]) - f(xint)))
    else:
      errloc = np.max(abs(np.dot(intpsi, Q[eltdofs]) - np.dot(intpsi, f[eltdofs])))
    errMax = max(errMax, errloc)

  return errMax
