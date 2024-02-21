"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

time stepping methods for solving ODEs or PDEs in semi-discrete form
"""

import numpy as np


class RungeKutta(object):
  """
  Runge-Kutta (SSP) schemes for time stepping in semi-discrete numerical methods
  for PDEs
  """

  def __init__(self, RHS, stages=1):
    self.RHS     = RHS
    self.stages  = stages

    if stages == 1:
      self.coeff = np.array([[1.0, 0.0, 0.0, 1.0]])
    elif stages == 2:
      self.coeff = np.array([[1.0, 0.0, 0.0, 1.0],
                             [0.5, 0.5, 0.0, 0.5]])
    elif stages == 3:
      self.coeff = np.array([[1.0    , 0.0    , 0.0, 1.0    ],
                             [3.0/4.0, 1.0/4.0, 0.0, 1.0/4.0],
                             [1.0/3.0, 2.0/3.0, 0.0, 2.0/3.0]])
    else:
      raise ValueError("This method has not been implemented yet!")


  def step(self, Q, t, dt):

    # initialize arrays for Runge-Kutta loop
    Q_last = np.copy(Q)
    Q_scnd = 0.0

    # Runge-Kutta loop
    for stage in range(self.stages):

      # compute next stage
      Q_next =      self.coeff[stage,0] * Q + \
                    self.coeff[stage,1] * Q_last + \
                    self.coeff[stage,2] * Q_scnd + \
               dt * self.coeff[stage,3] * self.RHS(Q_last, t)

      # update for next stage
      Q_last = np.copy(Q_next)

      # store the 2nd stage for SSP(5,3)
      if stage == 2:
        Q_scnd = np.copy(Q_next)

    return Q_next


class RungeKuttaDG(RungeKutta):
  """
  Runge-Kutta for DG schemes
  """

  def __init__(self, RHS, limiter, stages=1):

    super(RungeKuttaDG, self).__init__(RHS, stages)
    self.limiter = limiter


  def step(self, Q, Qold, t, dt):

    # initialize arrays for Runge-Kutta loop
    Q_last = np.copy(Q)
    Q_scnd = 0.0

    # Runge-Kutta loop
    for stage in range(self.stages):

      # compute next stage and apply limiter to it
      Q_next =      self.coeff[stage,0] * Q + \
                    self.coeff[stage,1] * Q_last + \
                    self.coeff[stage,2] * Q_scnd + \
               dt * self.coeff[stage,3] * self.RHS(Q_last, Qold, t)

      Q_next = self.limiter(Q_next)

      if (Q.shape[1]==4):
        Q_next[:,-1] = Q[:,-1]   # r=1
        #Q_next[:,-1] = 2.*Q[:,-1] - Qold[:,-1]   # r=2

      # update for next stage
      Q_last = np.copy(Q_next)

      # store the 2nd stage for SSP(5,3)
      if stage == 2:
        Q_scnd = np.copy(Q_next)

    return Q_next
