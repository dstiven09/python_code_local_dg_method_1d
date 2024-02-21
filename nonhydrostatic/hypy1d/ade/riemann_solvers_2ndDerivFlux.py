#"""
#hypy1d - hyperbolic differential equations with python in 1d
#Stefan Vater (2013)

#Riemann solvers for the shallow water equations
#"""

#import numpy as np
#from .equation_2ndDerivFlux import EqADE

#class EqADEUpWind(EqADE):
#  """
#  ADE object augmented by the Rusanov Riemann solver
 # """
#
 #   
  #def num_flux(self, ql, dql, qr, dqr):
   # """
    #evaluate numerical flux from left and right states ql and qr
    #"""
#
#    eps  = self.eps
#    flux = ql-eps*0.5*(dqr+dql)
#    
#    return flux
#    
#"""
#hypy1d - hyperbolic differential equations with python in 1d
#Stefan Vater (2013)
#
#Riemann solvers for the shallow water equations
#"""

import numpy as np
from .equation_2ndDerivFlux import EqADE

class EqADEUpWind(EqADE):
  """
  DE object augmented by the Rusanov Riemann solver
  """

    
  def num_fluxAV(self, ql, dql, qr, dqr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    eps  = self.eps
    flux = eps*0.5*(qr+ql)
    
    return flux
    
  def num_fluxAVderiv(self, ql, dql, qr, dqr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    eps  = self.eps
    flux = eps*0.5*(dqr+dql)
    
    return flux

  def num_fluxJump(self, ql, dql, qr, dqr):
    """
    evaluate numerical flux from left and right states ql and qr
    """

    eps  = self.eps
    flux = eps*(qr-ql)
    
    return flux