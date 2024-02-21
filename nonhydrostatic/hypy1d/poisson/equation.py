"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

functionality specific to the Poisson equation, using the Local DG method (LDG)
"""

import numpy as np


class Eqell:
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

  def __init__(self, DGElmt, Src, g, d, btopo, A=0., B=0., nhnl=1, nht=1, wettol=1.0e-8):
    self.DGEl     = DGElmt
    self.Src      = Src
    self.g        = g
    self.d        = d
    self.btopo    = btopo
    self.A        = A
    self.B        = B
    self.nhnl     = nhnl
    self.nht      = nht
    self.unknowns = 2
    self.wettol   = wettol

  def fact_f1_var2(self, qi):
    """
    factor of source term in first equation in front of second variable
    """

    return np.ones(len(qi[:,0]))

  def fact_s1_var1(self, qi, bi, dt):
    """
    factor of source term in first equation in front of second variable
    """

    return -np.ones(len(qi[:,0]))

  def fact_s1_var2(self, qi, dhi, dbi, bi):
    """
    factor of source term in first equation in front of second variable
    """

    return np.zeros(len(qi[:,0]))

  def fact_s1_c(self, qi, bi, dt):
    """
    source term in first equation on right hand side
    """

    return np.zeros(len(qi[:,0]))

  def fact_f2_var1(self, qi):
    """
    factor of source term in second equation in front of gradient of first variable
    """

    return np.ones(len(qi[:,0]))

  def fact_s2_var1(self, qi, dhi, bi, dbi, dt):
    """
    factor of source term in second equation in front of gradient of first variable
    """

    return np.zeros(len(qi[:,0]))

  def fact_s2_var2(self, qi, bi, dt):
    """
    factor of source term in second equation in front of gradient of first variable
    """

    return np.zeros(len(qi[:,0]))

  def fact_s2_c(self, qi, bi, ielmt):
    """
    source term in second equation on right hand side
    """

    return (self.Src.elmtsource(ielmt, qi, 1.))[:,1]

  def corr_hw(self, qi, hw, pnh, dt):
    """
    ...
    """
    return dt*(self.nhA*pnh/(qi[:,0])+self.nhB)+hw

  def corr_hflx(self, qi, hu, dt):
    """
    ...
    """

    return dt*(hu-qi[:,1])

  def hwpredict_flx(self, qi, hw, dt):
    """
    ...
    """

    ui = self.u(qi)

    return dt*hw*ui


  def u(self, qi):
    """
    compute velocity from state vector qi = (q, u), taking care of dry states
    """

    #if (qi[0] < self.wettol):
      #return 0.0
    #else:
      #return qi[1] / qi[0]
    return qi[1]

  def Flux(self, qi, iswet=1.0):
    """
    compute flux vector from state vector qi = (q, u)
    """

    ui = self.u(qi)

    return np.array([qi[1], qi[0]])

class EqellSource:
  """
  source term for Poisson equation
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

    #si = np.zeros(Qelmt.shape)
    si = np.zeros((Qelmt.shape[0],2))

    ## compute unknowns and their derivatives at quadrature points
    #f    = -np.sin(np.pi*(self.DGEl.dofcoordinates[self.DGEl.elementdofs[ielmt]]+1.))*(np.pi**2)
    f    = Qelmt[:,2]
    fi   = np.dot(self.Quad.psi.T, f)
    #qi   = ((np.dot(self.Quad.psi.T, Qelmt)).T)[0]
    #qi   = np.dot(Qelmt.T[0], self.Quad.psi)
    #qi2   = (np.dot(self.Quad.psi.T, Qelmt))[:,0] # different computation, should be a column vector, but is not!
    qi   = np.dot(self.Quad.psi.T, Qelmt[:,0]).T

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # add source term at quadrature points
      Src = np.array([-qi[iquad], fi[iquad]])

      for idof in range(self.DGEl.edofs):
        si[idof,:] = si[idof,:] + \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*Src

    return si

class AnaSol:
  """
  analytical solutions to dispersive equation set for different test cases
  """

  def __init__(self, DGElmt):
    self.DGEl     = DGElmt


  def analytical_poisson(self):
    """
    analytic solution of a general version of the Poisson problem
    """

    x = self.DGEl.dofcoordinates
    a = 1.
    Q = np.zeros((len(x),4))

    Q[:,0] = a*np.ones(len(x))                    # h: h
    Q[:,1] = np.cos(np.pi*(x+1.))*(np.pi)/a       # q: hu (analyt. sol. for a=const, not used)
    Q[:,2] = -np.sin(np.pi*(x+1.))*(np.pi**2)     # rhs in 2. equation: hw
    Q[:,3] = np.sin(np.pi*(x+1.))/a**2            # u: pnh ( analyt. sol. for a=const, not used)

    #Q[:,0] = np.cos(np.pi*(x+1.))*(np.pi)/a        # h: h
    #h_x    = -np.sin(np.pi*(x+1.))*(np.pi**2)/a
    #Q[:,1] = Q[:,0]**2                             # q: hu (analyt. sol. for a=const, not used)
    #Q[:,2] = 2.*Q[:,1]*h_x                           # rhs in 2. equation: hw
    #Q[:,3] = np.sin(np.pi*(x+1.))/a                # u: pnh ( analyt. sol. for a=const, not used)

    #Q[:,0] = (2.+np.cos(np.pi*(x+1.)))/a            # h: h
    #u_x    = np.cos(np.pi*(x+1.))/a*np.pi
    #u_xx   = -np.sin(np.pi*(x+1.))/a*(np.pi)**2
    #h_x    = -np.sin(np.pi*(x+1.))/a*(np.pi)
    #q_x    = Q[:,0]*u_xx + h_x*u_x
    #Q[:,1] = Q[:,0]*u_x                             # q: hu (analyt. sol. for a=const, not used)
    #Q[:,2] = Q[:,0]*q_x                             # rhs in 2. equation: hw
    #Q[:,3] = (0.+np.sin(np.pi*(x+1.)))/a            # u: pnh ( analyt. sol. for a=const, not used)

    #n = 2.
    #b = 1.
    #c = 0.

    #Q[:,0] = b*x+c                                 # h: h
    #Q[:,1] = 2.*a*x*(b*x+c)                        # q: hu (analyt. sol., not used)
    #Q[:,2] = -2.*a*(2.*(b**2)*(x**2)+3.*b*c*x+c**2)    # rhs in 2. equation: hw
    #Q[:,3] = a*(x**n-1.)                           # u: pnh ( analyt. sol. for a=const, not used)

    return Q

  def analytical_poisson_refl(self):
    """
    analytic solution of a general version of the Poisson problem
    """

    x = self.DGEl.dofcoordinates
    a = 1.
    Q = np.zeros((len(x),4))

    Q[:,0] = a*np.ones(len(x))                    # h: h
    Q[:,1] = -np.sin(np.pi*(x+1.))*(np.pi)/a      # q: hu (analyt. sol. for a=const, not used)
    Q[:,2] = -np.cos(np.pi*(x+1.))*(np.pi**2)     # rhs in 2. equation: hw
    Q[:,3] = np.cos(np.pi*(x+1.))/(a**2) -1.      # u: pnh ( analyt. sol. for a=const, not used)

    return Q