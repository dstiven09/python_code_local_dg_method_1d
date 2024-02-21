"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

finite volume limiter functions
"""

import numpy as np

def BFluxInternal(Ql, Qr, side, numFlux):
  """
  evaluation of the boundary flux

  flux evaluation is done by taking the internal value.

  Input values:
    Ql      : states on left side of each cell
    Qr      : states on right side of each cell
    side    : boundary on lower or upper side:
                side = 0 -> left boundary
                side = 1 -> right boundary
    numFlux : numerical flux function

  Return values:
    boundary flux

  See also:

  TODO: Might be worth to define this as a class to predefine the numerical flux.
  """

  if (side == 0):
    q = Ql[0]
  elif (side == 1):
    q = Qr[-1]
  else:
    raise ValueError('side must be either 0 or 1!')

  return numFlux(q, q)
