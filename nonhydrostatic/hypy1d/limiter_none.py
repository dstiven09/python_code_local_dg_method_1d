"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

Limiter for the shallow water equations
"""

import numpy as np


class Limiter:
  """
  wet/dry limiter
  """

  def __init__(self, Grid, Equation, Src, DGElmt):
    self.Gr    = Grid
    self.Eq    = Equation
    self.Src   = Src
    self.DGEl  = DGElmt


  def __call__(self, Q):
    """
    ...
    """

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):
      pass

    return Q
