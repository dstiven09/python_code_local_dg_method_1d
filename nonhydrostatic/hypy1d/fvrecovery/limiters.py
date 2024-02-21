"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

finite volume limiter functions
"""

import numpy as np

def LimCentered(a, b):
  """
  compute centered slopes (no limiter)

  Returns the mean of a and b which results in a centered difference when a
  and b are left and right differences of some function on a discrete grid.

  Input values:
    a : first slope
    b : second slope

  Return values:
    s : limited slope

  See also: LimMC, LimMinMod, LimSuperbee, LimVanLeer
  """

  return (a + b) / 2.0


def LimMC(a, b):
  """
  monotonized central-differences limiter

  Applies the monotonized central-differences (MC) limiter to left and right
  differences a and b of some function on a discrete grid.

  Input values:
    a : first slope
    b : second slope

  Return values:
    s : limited slope

  See also: LimCentered, LimMinMod, LimSuperbee, LimVanLeer
  """

  return (a * b > 0).astype(float) * np.sign(a) * \
         np.min(np.vstack((abs((a+b)/2.0), abs(2.0*a), abs(2.0*b))), axis=0)


def LimMinMod(a, b):
  """
  MinMod limiter

  Applies the MinMod limiter to left and right
  differences a and b of some function on a discrete grid.

  Input values:
    a : first slope
    b : second slope

  Return values:
    s : limited slope

  See also: LimCentered, LimMC, LimSuperbee, LimVanLeer
  """

  return (a * b > 0).astype(float) * np.sign(a) * np.min(np.vstack((abs(a), abs(b))), axis=0)


def LimSuperbee(a, b):
  """
  Superbee limiter

  Applies the Superbee limiter to left and right
  differences a and b of some function on a discrete grid.

  Input values:
    a : first slope
    b : second slope

  Return values:
    s : limited slope

  See also: LimCentered, LimMC, LimMinMod, LimVanLeer
  """

  s1 = LimMinMod(a, 2.0*b)
  s2 = LimMinMod(2.0*a, b)
  return np.sign(s1) * np.max(np.vstack((abs(s1), abs(s2))), axis=0)


def LimVanLeer(a, b):
  """
  van Leer limiter

  Applies the van Leer limiter to left and right
  differences a and b of some function on a discrete grid.

  Input values:
    a : first slope
    b : second slope

  Return values:
    s : limited slope

  See also: LimCentered, LimMC, LimMinMod, LimSuperbee
  """

  s = np.zeros(a.shape)

  for i in range(a.size):
    if ((a[i] + b[i]) == 0):
      s[i] = 0.0
    else:
      s[i] = (a[i] * b[i] + abs(a[i] * b[i])) / (a[i] + b[i])

  return s
