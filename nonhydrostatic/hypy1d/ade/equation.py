"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Susanne Beckers (2015)

functionality specific to the advection equation
h_t + h_x = 0
and the advection  diffusion equation
h_t + h_x + eps*h_xx = 0
"""

import numpy as np
import math


class EqADE:
  """
  advection diffusion equations object
  """

  def __init__(self, g=1.0, eps=0.0):
    self.g        = g
    self.unknowns = 1
    self.eps      = eps


  def EVals(self, qi):
    """
    compute Eigen values from state vector qi = (h)
    """
    ci    = np.zeros(len(qi))
    ci[:] = 1.0
    return ci


  def LEVec(self, qi):
    """
    compute left Eigen vectors from state vector qi = (h)
    """
    ci    = np.zeros(len(qi))
    ci[:] = 1.0
    return ci


  def REVec(self, qi):
    """
    compute right Eigen vectors from state vector qi = (h)
    """
    ci    = np.zeros(len(qi))
    ci[:] = 1.0
    return ci


  def Flux(self, qi, iswet=1.0):
    """
    compute flux vector from state vector qi = (h)
    """
    return qi


  def DFlux(self, qi, dqi, iswet=1.0):
    """
    compute spatial derivative of flux vector (flux divergence) from state
    vector qi = (h), and its derivative dqi
    """
    return dqi


class EqAESource:
  """
  source term for advection equation (DG discretization)
  """

  def __init__(self, Equation, DGElmt, Quad, eps):
    self.Eq    = Equation
    self.DGEl  = DGElmt
    self.Quad  = Quad
    self.eps   = 0.0

    self.btopo = np.zeros(DGElmt.doflength)

  def update(self, btopo):
    """
    update/initialize source term
    """
    self.btopo = btopo

  def elmtsource(self, ielmt, Qelmt, t):
    """
    compute source term for one element, here source is zero
    """

    #si = np.zeros((2,1))
    si = np.zeros(Qelmt.shape)

    return si


class EqADESource:

  def __init__(self, Equation, DGElmt, Quad, eps):
    self.Eq    = Equation
    self.DGEl  = DGElmt
    self.Quad  = Quad
    self.eps   = eps

    self.btopo = np.zeros(DGElmt.doflength)


  def update(self, btopo):
    """
    update/initialize source term
    """
    self.btopo = btopo

  def elmtsource(self, ielmt, Qelmt, t):
    """
    compute source term for one element, here source is diffusion term
    """

    #si = np.zeros((2,1))
    si = np.zeros(Qelmt.shape)

    # compute unknowns and their second derivatives at quadrature points
    ddxxq = np.dot(self.DGEl.ddxx[ielmt], self.Quad.psi)
    qi    = np.dot(self.Quad.psi.T, Qelmt[:,0])
    ddqi  = np.dot(ddxxq.T, qi)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # add source term at quadrature points
      Src = np.array(ddqi[iquad])

      for idof in range(self.DGEl.edofs):
        si[idof] = si[idof] - \
          self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*self.eps*Src

    return si
    
class AnaSolution:
    
 def __init__(self, t, eps):
    self.t    = t
    self.eps  = eps

        
 def AnalyticDiscrete(self, x):
     t   = self.t
     eps = self.eps
     l   = x.shape[0]
     f   = np.zeros(l)
     for i in range(l):
         if (eps == 0.0):
             if ((x[i]-t>=-1) and (x[i]-t<=0)):
                f[i] = 1.0
             else:
                f[i] = 0.0
         else:
             f[i]= 0.5*(math.erf((x[i]+1-t)/(2*math.sqrt(t*eps)))- \
                 math.erf((x[i]-t)/(2*math.sqrt(t*eps))))
             #f[i]= 0.5*(math.erf((x[i]+1-t)/(2*math.sqrt(t*eps)))- \
             #    math.erf((x[i]-t)/(2*math.sqrt(t*eps))))+\
             #   0.5*(math.erf((x[i]+3-t)/(2*math.sqrt(t*eps)))- \
             #    math.erf((x[i]+2-t)/(2*math.sqrt(t*eps)))) 
         ##f[i]=math.sin(3.1415926535*(x[i]-t))
       # if (x[i]-t>=-1):
       #     f[i]=(x[i]-t)**4-2*(x[i]-t)**2+1
       # else:
       #     f[i]=(x[i]-t+2)**4-2*(x[i]-t+2)**2+1
        
     return f
     
 def AnalyticConti(self, x):
     t   = self.t
     eps = self.eps
     f = np.zeros(x.shape)
     if (eps == 0.0):
         if ((x-t>=-1) and (x-t<=0)):
           f = 1.0
         else:
           f = 0.0
     else:
         f = 0.5*(math.erf((x+1-t)/(2*math.sqrt(t*eps)))- \
                 math.erf((x-t)/(2*math.sqrt(t*eps))))
        #f = 0.5*(math.erf((x+1-t)/(2*math.sqrt(t*eps)))- \
        #         math.erf((x-t)/(2*math.sqrt(t*eps))))+\
        #         0.5*(math.erf((x+3-t)/(2*math.sqrt(t*eps)))- \
        #         math.erf((x+2-t)/(2*math.sqrt(t*eps))))
                 
     ##f=math.sin(3.1415926535*(x-t))
     #if (x-t>=-1):
     #   f=(x-t)**4-2*(x-t)**2+1
     #else:
     #    f=(x-t+2)**4-2*(x-t+2)**2+1
         
     return f
     