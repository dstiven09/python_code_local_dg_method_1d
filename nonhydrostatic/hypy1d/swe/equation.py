"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

functionality specific to the shallow water equations
"""

import numpy as np


class EqSWE:
  """
  shallow water equations object
  """

  def __init__(self, Localnh, d, g=1.0, swnl=1, nht=0, A=0., B=0., wettol=1.0e-8, numtracer=0, phitol=1.0e-8):
    self.d        = d
    self.g        = g
    self.unknowns = 2+numtracer
    self.swnl     = swnl
    self.nht      = nht
    self.nhA      = A
    self.nhB      = B
    self.wettol   = wettol
    self.phitol   = phitol
    self.Local    = Localnh


  def u(self, qi):
    """
    compute velocity from state vector qi = (h, hu), taking care of dry states
    """

    if (qi[0] < self.wettol):
      return 0.0
    else:
      return qi[1] / qi[0]


  def phii(self, qi, i):
    """
    compute ith tracer from state vector qi = (h, hu, phi1, phi2, ...), taking care of dry states

    todo: check i <= numtracer
    """

    if (qi[0] < self.phitol):
      return 0.0
    else:
      return qi[i+2] / qi[0]


  def EVals(self, qi):
    """
    compute Eigen values from state vector qi = (h, hu)
    """

    ci = np.sqrt(self.g * qi[0])
    ei = np.ones(self.unknowns) * self.u(qi)
    ei[ 0] = ei[ 0] - ci
    ei[-1] = ei[-1] + ci

    return ei


  def LEVec(self, qi):
    """
    compute left Eigen vectors from state vector qi = (h, hu)

    todo: adjust for numtracer > 0
    """

    ci = np.sqrt(self.g * qi[0])
    ui = self.u(qi)

    return np.array([[ ui + ci, -1],
                     [-ui + ci,  1]])


  def REVec(self, qi):
    """
    compute right Eigen vectors from state vector qi = (h, hu)

    todo: adjust for numtracer > 0
    """

    ci = np.sqrt(self.g * qi[0])
    ui = self.u(qi)

    return np.array([[1      , 1      ],
                     [ui - ci, ui + ci]])


  def Flux(self, qi, iswet=1.0):
    """
    compute flux vector from state vector qi = (h, hu)
    """

    #ci = np.sqrt(self.g * qi[0])
    ui = self.u(qi)
    fi = np.zeros(self.unknowns)
    fi[0] = qi[1]
    if (self.swnl==0):
      fi[1] = iswet*self.g*qi[0]*self.d
    if (self.swnl==2):
      fi[1] = iswet*self.g*qi[0]**2 / 2.0
    if (self.swnl==1):
      fi[1] = qi[1] * ui + iswet*self.g*qi[0]**2 / 2.0
    for i in range(2,self.unknowns):
      if (self.swnl==1):
        fi[i] = qi[i] * ui

    return fi


  def FluxGrav(self, qi):
    """
    compute flux vector from state vector qi = (h, hu)
    """

    fi = np.zeros(self.unknowns)
    if ((self.swnl==1)|(self.swnl==2)):
      fi[1] = self.g*qi[0]**2 / 2.0
    if (self.swnl==0):
      fi[1] = self.g*qi[0]*self.d

    return fi


  def DFlux(self, qi, dqi, iswet=1.0):
    """
    compute spatial derivative of flux vector (flux divergence) from state
    vector qi = (h, hu), and its derivative dqi
    """

    ui  = self.u(qi)
    dfi = np.zeros(self.unknowns)
    dfi[0] = dqi[1]
    if (self.swnl==0):
      dfi[1] = iswet*self.g*self.d*dqi[0]
    if (self.swnl==2):
      dfi[1] = iswet*self.g*qi[0]*dqi[0]
    if (self.swnl==1):
      dfi[1] = -ui**2*dqi[0] + 2.0*ui*dqi[1] + iswet*self.g*qi[0]*dqi[0]
    for i in range(2,self.unknowns):
      if (self.swnl==1):
        phi = phii(qi,i-2)
        dfi[i] = phi*dqi[1] + ui*dqi[i] - ui*phi*dqi[0]

    return dfi


  def DFluxAdv(self, qi, dqi):
    """
    compute spatial derivative of self.nhADVECTIVE part of flux vector (flux divergence) from state
    vector qi = (h, hu), and its derivative dqi
    """

    ui  = self.u(qi)
    dfi = np.zeros(self.unknowns)
    dfi[0] = dqi[1]
    if (self.swnl==1):
      dfi[1] = -ui**2*dqi[0] + 2.0*ui*dqi[1]
      for i in range(2,self.unknowns):
        phi = phii(qi,i-2)
        dfi[i] = phi*dqi[1] + ui*dqi[i] - ui*phi*dqi[0]

    return dfi

class EqSWESource:
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

  def multruns_input(self):

    value  = np.loadtxt('multruns_in')

    return value

  def elmtsource(self, ielmt, Qelmt, Qoldelmt, t):
    """
    compute source term for one element
    """

    si = np.zeros(Qelmt.shape)

    # compute unknowns and their derivatives at quadrature points
    ddxq = np.dot(self.DGEl.ddx[ielmt], self.Quad.psi)
    qi   = np.dot(self.Quad.psi.T, Qelmt)
    dbi  = np.dot(ddxq.T, self.btopo[self.DGEl.elementdofs[ielmt]])

    # compute terms for linearized equations
    bi   = np.dot(self.Quad.psi.T, self.btopo[self.DGEl.elementdofs[ielmt]])
    dqbl = np.dot(ddxq.T, Qelmt[:,0]+self.btopo[self.DGEl.elementdofs[ielmt]])

    if (np.intersect1d(np.array([ielmt]), self.Eq.Local.elmtloc)==ielmt):
      iel = 1
    else:
      iel = 0

    if ((self.Eq.nht==2) and (iel==1)):
      eltdofs = self.DGEl.elementdofs[ielmt]
      if (self.Eq.swnl>0):
        h     = Qoldelmt[:,0]
      if (self.Eq.swnl==0):
        h    = self.Eq.d - self.btopo[eltdofs]

      pnh    = Qoldelmt[:,3]
      dhpnhi = np.dot(ddxq.T, h*pnh)
      pnhi   = np.dot(self.Quad.psi.T, pnh)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):

      # add source term at quadrature points
      Src = np.zeros(Qelmt.shape[1])
      if ((self.Eq.swnl==1)|(self.Eq.swnl==2)):
        Src[1] = self.Eq.g * qi[iquad,0] * dbi[iquad]
        if ((self.Eq.nht==2) and (iel==1)):
          Src[1] = Src[1] + dhpnhi[iquad] + (self.Eq.nhA*pnhi[iquad]+self.Eq.nhB)*dbi[iquad]
          Src[2] = Src[2] - pnhi[iquad]*self.Eq.nhA-self.Eq.nhB
      if (self.Eq.swnl==0):
        #Src[1] = self.Eq.g * (-bi[iquad]*dqi0[iquad] + (self.Eq.d-bi[iquad]) * dbi[iquad])
        Src[1] = self.Eq.g * (-bi[iquad]*dqbl[iquad] + self.Eq.d * dbi[iquad])
        if ((self.Eq.nht==2) and (iel==1)):
          Src[1] = Src[1] + dhpnhi[iquad] + (self.Eq.nhA*pnhi[iquad]+self.Eq.nhB)*dbi[iquad]
          Src[2] = Src[2] - pnhi[iquad]*self.Eq.nhA-self.Eq.nhB

      for idof in range(self.DGEl.edofs):
        si[idof,:] = si[idof,:] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*Src

    return si

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
    pnh = hpnh/h

    return h, hu, hw, pnh

class EqSWEFVSource:
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
    src[:,1] = self.Eq.g / 2.0 * ((Qr[:,0]**2 - hr**2) + (hl**2 - Ql[:,0]**2) + (hl + hr) * (bl - br)) / self.gr.elementwidth

    return src
