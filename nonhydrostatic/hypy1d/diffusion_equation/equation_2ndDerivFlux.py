# -*- coding: utf-8 -*-
"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Susanne Beckers (2015)

functionality specific to the diffusion equation
h_t - eps*h_xx = f
"""



import numpy as np
import math


class EqDE:
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
    ci[:] = 0.0
    return ci


  def LEVec(self, qi):
    """
    compute left Eigen vectors from state vector qi = (h)
    """
    ci    = np.zeros(len(qi))
    ci[:] = 0.0
    return ci


  def REVec(self, qi):
    """
    compute right Eigen vectors from state vector qi = (h)
    """
    ci    = np.zeros(len(qi))
    ci[:] = 0.0
    return ci


  def Flux(self, qi, dqi):
    """
    compute flux vector from state vector qi = (h)
    """
    eps = self.eps
    return -eps*dqi


  def DFlux(self, qi, dqi, ddqi):
    """
    compute spatial derivative of flux vector (flux divergence) from state
    vector qi = (h), and its derivative dqi
    """
    eps = self.eps
    return -eps*ddqi


class EqSource:

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

    
  def elmtsource(self, ielmt, elmtcoor, Qelmt, t):
    """
    compute source term for one element, here source is diffusion term
    """

    si = np.zeros(Qelmt.shape)
    Src = np.zeros(Qelmt.shape)

    # compute unknowns and their second derivatives at quadrature points
    ddxxq = np.dot(self.DGEl.ddxx[ielmt], self.Quad.psi)
    qi    = np.dot(self.Quad.psi.T, Qelmt[:,0])
    ddqi  = np.dot(ddxxq.T, qi)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # add source term at quadrature points
        # Reviere example
          # see in loop!
          
        # x⁴ equation
          #e = 100.0
          #Src = 12*e*np.multiply(elmtcoor,elmtcoor)#-np.array(ddqi[iquad])
        # x² equation
          Src = -2

          for idof in range(self.DGEl.edofs):
              #Src[idof] = math.cos(8.0*math.pi*elmtcoor[idof])
              si[idof] = si[idof] - \
                 self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*Src

    return si
    
    
class AnaSolution:
#================================================================= 
#==================================================================     
    #"""
    #For sine test case: u_t-eps*u_xx= -12*x^2
    #    u(x,0) = sin(-1/eps*x)+cos(-1/eps*x)
    #    u(x,0) = 1/eps*x^4-2/eps*x^2+1
    #"""
    
 def __init__(self, t, eps):
    self.t    = t
    self.eps  = eps

        
 def AnalyticDiscrete(self, x):
     t   = self.t
     eps = self.eps
     l   = x.shape[0]
     f   = np.zeros(l)
     e    = 100.0
     
     for i in range(l):
       # Reviere example
         #f[i] = math.cos(8.0*math.pi*x[i])
       # x⁴ function
         #f[i]= 1.0*e/eps*x[i]**4.0-2.0*e/eps*x[i]**2.0+e -\
         #      4.0*e*t
       # x² function
         f[i]= -(x[i]+1)*(x[i]-1)
   
     return f
     
 def AnalyticConti(self, x):
     t   = self.t
     eps = self.eps
     # Reviere example
        #f = math.cos(8.0*math.pi*x)
     # x⁴ function
        #e    = 100.0
        #f = 1.0*e/eps*x**4.0-2.0*e/eps*x**2.0+e-4.0*e*t
     # x² function
     f=-(x-1)*(x+1)   
            
     return f
  
 def AnalyticDerivative(self, x):
     eps = self.eps
     # Reviere example
        #df = -8.0*math.pi*math.sin(8.0*math.pi*x)
     # x⁴ function
        #e = 100.0
        #df = 4.0*e/eps*x**3.0-4.0*e/eps*x
     # x² function
     df = -2*x
     
     return df
     