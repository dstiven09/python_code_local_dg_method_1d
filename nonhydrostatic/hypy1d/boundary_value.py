"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

definition of boundary values
"""

import numpy as np


class BoundaryValue(object):
  """
  base class for definition of boundary value
  """

  def __init__(self, Grid, bnode):

    self.Gr   = Grid
    self.bnode = bnode

    # Note: here we assume that the first entry in nodeelements is the left
    #       element, and the second entry is the right one
    if (self.Gr.nodeelements[self.bnode,0]<0):
      self.side = 0
    elif (self.Gr.nodeelements[self.bnode,1]<0):
      self.side = 1
    else:
      raise ValueError("This is not a valid boundary!")


class BoundaryValueFVZeroExtrap(BoundaryValue):
  """
  boundary value for FV discretization based on zeroth order extrapolation
  """

  def __call__(self, Q, t):
    """
    evaluate boundary value from state Q at time t
    """

    ndelts = self.Gr.nodeelements[self.bnode]
    if (self.side==0):
      return Q[ndelts[1]]
    else:
      return Q[ndelts[0]]


class BoundaryValueDG(BoundaryValue):
  """
  base class for definition of boundary value in DG context
  """

  def __init__(self, Grid, bnode, DGElmt):

    super(BoundaryValueDG, self).__init__(Grid, bnode)
    self.DGEl = DGElmt


class BoundaryValueDGZeroExtrap(BoundaryValueDG):
  """
  boundary value for DG discretization based on zeroth order extrapolation
  """

  def __call__(self, Q, t):
    """
    evaluate boundary value from state Q at time t
    """

    ndelts = self.Gr.nodeelements[self.bnode]
    if (self.side==0):
      erdofs = self.DGEl.elementdofs[ndelts[1],:]
      return Q[erdofs[0]]
    else:
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      return Q[eldofs[-1]]

class BoundaryValueDGWall(BoundaryValueDG):
  """
  boundary value for DG discretization implementing wall b.c. for SWE
  """

  def __call__(self, Q, t):
    """
    evaluate boundary value from state Q at time t
    """

    ndelts = self.Gr.nodeelements[self.bnode]
    if (self.side==0):
      erdofs = self.DGEl.elementdofs[ndelts[1],:]
      Qtmp = Q[erdofs[0]]
    else:
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      Qtmp = Q[eldofs[-1]]

    Qtmp[1] = -Qtmp[1]

    return Qtmp


class BoundaryValueDGInflowOutflow(BoundaryValueDG):
  """
  boundary value for DG discretization, inflow/outflow for SWE
  """

  def __init__(self, Grid, bnode, DGElmt, grav, h0, u0, sshfunc, unknowns=2, tracfunc=None):

    super(BoundaryValueDGInflowOutflow, self).__init__(Grid, bnode, DGElmt)
    self.g    = grav
    self.h0   = h0
    self.u0   = u0
    self.ssh  = sshfunc
    self.unkn = unknowns
    self.trac = tracfunc
    self.sqrtgh0 = np.sqrt(self.g*self.h0)

  def __call__(self, Q, t):
    """
    evaluate boundary value from state Q at time t
    """

    hp  = self.ssh(t)
    hup = 2.0*(hp+self.h0)*(np.sqrt(self.g*(hp+self.h0))-self.sqrtgh0)

    #R = np.zeros((self.unkn, self.unkn))
    #R[0 ,:self.unkn] = [1.0, self.u0-self.sqrtgh0]
    #R[-1,:self.unkn] = [1.0, self.u0+self.sqrtgh0]

    #if(self.unkn == 3):
      #R[0,2] = 0.0
      #R[1,:] = [0.0, 0.0, 1.0]

    r1 = [1.0, self.u0-self.sqrtgh0]
    r2 = [1.0, self.u0+self.sqrtgh0]

    ndelts = self.Gr.nodeelements[self.bnode]
    if (self.side==0):
      erdofs = self.DGEl.elementdofs[ndelts[1],:]
      Qi     = Q[erdofs[0]]
      w1 = 1.0/(2.0*self.sqrtgh0)*(( self.u0+self.sqrtgh0)*(Qi[0]-self.h0) - (Qi[1]-self.h0*self.u0))
      w2 = 1.0/(2.0*self.sqrtgh0)*((-self.u0+self.sqrtgh0)*hp              + hup)
    else:
      eldofs = self.DGEl.elementdofs[ndelts[0],:]
      Qi     = Q[eldofs[-1]]
      w1 = 1.0/(2.0*self.sqrtgh0)*(( self.u0+self.sqrtgh0)*hp              + hup)
      w2 = 1.0/(2.0*self.sqrtgh0)*((-self.u0+self.sqrtgh0)*(Qi[0]-self.h0) + (Qi[1]-self.h0*self.u0))

    Qb     = np.zeros(Qi.shape)
    #Qb     = np.zeros((Qi.shape[0],self.unkn))  #change to include P[:,1] into Q
    Qb[0]  = self.h0         + w1*r1[0] + w2*r2[0]
    Qb[1]  = self.h0*self.u0 + w1*r1[1] + w2*r2[1]

    if (self.unkn == 3):
      #Qb[2] = self.trac(t)
      Qb[2]  = self.h0*self.u0 + w1*r1[1] + w2*r2[1]

    if (Qi.shape[0] == self.unkn+1):
      Qb[-1] = Qi[-1]

    return Qb
