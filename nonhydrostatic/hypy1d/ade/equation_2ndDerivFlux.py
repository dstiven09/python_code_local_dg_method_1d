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


  def Flux(self, qi, dqi):
    """
    compute flux vector from state vector qi = (h)
    """
    eps = self.eps
    return qi-eps*dqi


  def DFlux(self, qi, dqi, ddqi):
    """
    compute spatial derivative of flux vector (flux divergence) from state
    vector qi = (h), and its derivative dqi
    """
    eps = self.eps
    return dqi-eps*ddqi


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

  def elmtsource(self, ielmt,elmtcoor, Qelmt, t):
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

  #def elmtsource(self, ielmt, Qelmt, t):
  #  """
  #  compute source term for one element, here source is diffusion term
  #  """

  #  #si = np.zeros((2,1))
  #  si = np.zeros(Qelmt.shape)

  #  return si
    
  def elmtsource(self, ielmt, elmtcoor, Qelmt, t):
    """
    compute source term for one element, here source is diffusion term
    """

    si = np.zeros(Qelmt.shape)

    # compute unknowns and their second derivatives at quadrature points
    ddxxq = np.dot(self.DGEl.ddxx[ielmt], self.Quad.psi)
    qi    = np.dot(self.Quad.psi.T, Qelmt[:,0])
    ddqi  = np.dot(ddxxq.T, qi)

    # quadrature loop
    for iquad in range(self.Quad.quadpts):
      # add source term at quadrature points
      Src1 = np.array(qi[iquad])
      Src = np.array(ddqi[iquad])

      for idof in range(self.DGEl.edofs):
        si[idof] = si[idof] - \
           self.Quad.w[iquad]*self.Quad.eMinvpsi[idof, iquad]*self.eps*Src
        #si[idof] = self.Quad.w[iquad]*self.Quad.eMinvpsi[idof,iquad]* \
        #           (Src1-self.eps*Src2)
        #si = 0.0

    return si
    
    
class AnaSolutionStep:
    
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
             if ((x[i]-t>-1) and (x[i]-t<=0)):
                f[i] = 1.0
             else:
                f[i] = 0.0
         else:
             f[i]= 0.5*(math.erf((x[i]+1-t)/(2*math.sqrt(t*eps)))- \
                 math.erf((x[i]-t)/(2*math.sqrt(t*eps))))
     return f
     
 def AnalyticConti(self, x):
     t   = self.t
     eps = self.eps
     #f = np.zeros(x.shape)
     if (eps == 0.0):
         if ((x-t>=-1) and (x-t<=0)):
           f = 1.0
         else:
           f = 0.0
     else:
         f = 0.5*(math.erf((x+1-t)/(2*math.sqrt(t*eps)))- \
                 math.erf((x-t)/(2*math.sqrt(t*eps))))
     return f
  
 def AnalyticDerivative(self, x):
     t   = self.t
     eps = self.eps
     df = np.zeros(x.shape)
     #if (eps == 0.0) or (t == 0.0):
     df = 0.0
     #else:
         #f = 0.5/(math.sqrt(t*eps*math.pi))(math.exp((x+1-t)**2/(4*t*eps))- \
         #         math.exp((x-t)**2/(4*t*eps)))
     return df
#==================================================================

class Residual:

  def __init__(self, Grid, DGElmt, Quad, Quad2, eps):
      
    self.DGElmt = DGElmt
    self.Quad   = Quad
    self.Quad2  = Quad2
    self.Grid   = Grid
    self.eps    = eps
    
  def z (self, x, t, T):
      """
      Dual solution of advection diffusion equation
        u_t+u_x-eps*u_xx =0
      J(u) = =\int_\R u^2(x-1,T)-u^2(x-1,0) dx
      """
      if (T>t):
        z = 0.5*math.erf((1-x-T+t)/(2*math.sqrt((T-t)*self.eps)))-\
             0.5*math.erf((-x-T+t)/(2*math.sqrt((T-t)*self.eps)))
      else:  # discontinous dual initial conditions
          if (x>1) or (x<0):
              z = 0.0
          elif (x==0) or (x==1):
              z = 0.5
          else:
              z = 0.0
      return z
  
  def zx (self, x, t, T):
      """
      Spatial derivative of dual solution of advection diffusion equation
        u_t+u_x-eps*u_xx =0
      J(u) = =\int_\R u^2(x-1,T)-u^2(x-1,0) dx
      """
      if (T>t):
        zx = 1/math.sqrt(4*math.pi*self.eps*(T-t))*\
                (math.exp(-(1-x-T+t)**2/(4*(T-t)*self.eps))-\
                math.exp(-(-x-T+t)**2/(4*(T-t)*self.eps)))
      else:  # derivative of discontinous dual initial conditions = 0 a.e.
         zx = 0.0
      return zx
  
  def zt (self, x, t, T):
      """
      Spatial derivative of dual solution of advection diffusion equation
        u_t+u_x-eps*u_xx =0
      J(u) = =\int_\R u^2(x-1,T)-u^2(x-1,0) dx
      """
      if (T>t):
        zt = 1/math.sqrt(math.pi*self.eps*(T-t))*\
                ((1/math.sqrt(4*(T-t))-(1-x-T+t)/(4*math.sqrt((T-t)**3)))*\
                  math.exp(-(1-x-T+t)**2/(4*(T-t)*self.eps))-\
                (1/math.sqrt(4*(T-t))-(-x-T+t)/(4*math.sqrt((T-t)**3)))*\
                  math.exp(-(-x-T+t)**2/(4*(T-t)*self.eps)))
      else:  # erivative of discontinous dual initial conditions = 0 a.e. ???
         zt = 0.0
      return zt
    
  def WeightedResidual(self, Q, t, T):  
   """
   compute residual of a DG discretization with weight with dual solution z
   at time t
   """
   IQsq = 0.0

   for ielt in range(self.DGElmt.elementdofs.shape[0]):
    eltdofs = self.DGElmt.elementdofs[ielt]
    eltcoo  = self.Grid.nodecoordinates[self.Grid.elementnodes[ielt]]
    deltax  = eltcoo[1]-eltcoo[0]
    xq = (self.Quad.x+1.0)/2.0 * deltax + eltcoo[0]
    Z  = np.zeros(xq.shape[0])
    for i in range(xq.shape[0]):
         Z[i] = self.zt(xq[i],t,T)+self.zx(xq[i],t,T)
    QHelp  = np.dot(self.Quad.psi.T, Q[eltdofs])
    Qqsq   = np.multiply(QHelp[:,0],Z)
    IntQsq = self.DGElmt.J[ielt] * np.dot(self.Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq
    
   IQsq = IQsq - Q[self.Grid.nlength-1,0]*self.z(1,t,T)+ Q[0,0]*self.z(-1,t,T)

   return IQsq
  
  def WeightedB(self, Q, t, T):
    """
    Derivative of J(f)=\int_{-1}^1 f^2(x-1,t) dx
    in direction f-Q
    f analytic solution, Q numeric approximation
    """
    IQsq = 0.0

    for ielt in range(self.DGElmt.elementdofs.shape[0]):
      eltdofs = self.DGElmt.elementdofs[ielt]
      eltcoo  = self.Grid.nodecoordinates[self.Grid.elementnodes[ielt]]
      deltax  = eltcoo[1]-eltcoo[0]
      xq = self.Quad.x/2.0 * deltax + eltcoo[0]
      F  = np.zeros(xq.shape[0])
      for i in range(xq.shape[0]):
           F[i] = self.z(xq[i]-1,t,T)
      QHelp  = np.dot(self.Quad.psi.T, Q[eltdofs])
      Qqsq   = np.multiply(F,QHelp[:,0])

      #Integration
      IntQsq = self.DGElmt.J[ielt] * np.dot(self.Quad.w, Qqsq)
      IQsq   = IQsq + IntQsq
    return IQsq
