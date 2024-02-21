"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

Riemann solvers for the shallow water equations
"""

import numpy as np

from .equation import EqSWE


class EqSWERusanov(EqSWE):
  """
  SWE object augmented by the Rusanov Riemann solver
  """

  def num_flux(self, ql, qr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    ul = self.u(ql)
    ur = self.u(qr)
    
    # compute maximal flux strength
    lmaxl = abs(ul) + np.sqrt(self.g*abs(ql[0]))
    lmaxr = abs(ur) + np.sqrt(self.g*abs(qr[0]))
    lmax  = max(lmaxl, lmaxr)

    # flux evaluation
    return 0.5 * (self.Flux(ql) + self.Flux(qr) - lmax * (qr - ql))

  def num_fluxAdv(self, ql, qr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    # compute maximal flux strength
    lmax  = max(abs(self.u(ql)), abs(self.u(qr)))

    # flux evaluation
    return 0.5 * (self.FluxAdv(ql) + self.FluxAdv(qr) - lmax * (qr - ql))


class EqSWEHLLE(EqSWE):
  """
  SWE object augmented by the HLLE Riemann solver
  """

  def num_flux(self, ql, qr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    ul = self.u(ql)
    ur = self.u(qr)
    hm = (ql[0] + qr[0]) / 2.0

    # take care of dry states
    if hm < self.wettol:
      return self.Flux(ql)
    else:
      # compute Roe linearisation
      Sqhl = np.sqrt(ql[0])
      Sqhr = np.sqrt(qr[0])
      um   = Sqhl / (Sqhl + Sqhr) * ul + Sqhr / (Sqhl + Sqhr) * ur

      cl = np.sqrt(self.g * ql[0])
      cr = np.sqrt(self.g * qr[0])
      cm = np.sqrt(self.g * hm)

      # get minimum and maximum speeds of information spread
      # according to Einfeldt (1988)
      if ql[0] < self.wettol:
        bminus = min(ur-2.0*cr, 0.0)
      else:
        bminus = min(um-cm, ul-cl, 0.0)

      if qr[0] < self.wettol:
        bplus = max(ul+2.0*cl, 0.0)
      else:
        bplus = max(um+cm, ur+cr, 0.0)

      # flux evaluation
      return (bplus * self.Flux(ql) - bminus * self.Flux(qr) + \
              bplus * bminus * (qr - ql)) / (bplus - bminus)
