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

  def __init__(self, Grid, Equation, Src, DGElmt, wettol=1.0e-8):
    self.Gr     = Grid
    self.Eq     = Equation
    self.Src    = Src
    self.DGEl   = DGElmt
    self.wettol = wettol

    self.fin = np.finfo(float)


  def __call__(self, Q):
    """
    tbw
    """

    Qlim = np.copy(Q)
    H    = Q[:,0] + self.Src.btopo

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      elldofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,0]]
      elrdofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,1]]

      # limit in hydrostatic variable (h+b)
      dHe = (H[eltdofs[1]] - H[eltdofs[0]]) / 2.0
      Hem = (H[eltdofs[1]] + H[eltdofs[0]]) / 2.0
      Hlm = (H[elldofs[1]] + H[elldofs[0]]) / 2.0
      Hrm = (H[elrdofs[1]] + H[elrdofs[0]]) / 2.0

      dHmax = np.maximum(np.maximum(Hlm, Hem), Hrm) - Hem
      dHmin = Hem - np.minimum(np.minimum(Hlm, Hem), Hrm)
      dH    = np.sign(dHe) * np.minimum(abs(dHe), np.minimum(dHmax, dHmin))
      #dH    = np.sign(dHe) * np.minimum(abs(dHe), np.minimum(abs(Hrm-Hem), abs(Hem-Hlm)))
      #if (Hem-Hlm)*(Hem-Hrm)>=0.0:
        #dH = 0.0
      corr  = dH - dHe
      if(np.min(self.Gr.elementneighbors[ielmt]) > -1):
        Qlim[eltdofs[0],0] = Qlim[eltdofs[0],0] - corr
        Qlim[eltdofs[1],0] = Qlim[eltdofs[1],0] + corr

      helt  = Qlim[eltdofs,0]
      ihmin = np.argmin(helt)
      if (helt[ihmin] < 0.0):
        helt[1-ihmin]   = helt[1-ihmin] + helt[ihmin]
        helt[ihmin]     = 0.0
        Qlim[eltdofs,0] = helt

      # try to correct some round off errors
      #if (Qlim[eltdofs[0],0] < self.fin.eps):
        #Qlim[eltdofs[0],0] = 0.0
      #if (Qlim[eltdofs[1],0] < self.fin.eps):
        #Qlim[eltdofs[1],0] = 0.0

      # check for positivity of mean height
      if (Qlim[eltdofs[0],0]+Qlim[eltdofs[1],0] < 0.0):
        print(Qlim[eltdofs,0])
        print self.DGEl.dofcoordinates[self.DGEl.elementdofs[ielmt]]
        raise ValueError

      # limit momentum based on velocity
      dme = (Q[eltdofs[1],1] - Q[eltdofs[0],1]) / 2.0
      Qem = (Q[eltdofs[1]] + Q[eltdofs[0]]) / 2.0
      Qlm = (Q[elldofs[1]] + Q[elldofs[0]]) / 2.0
      Qrm = (Q[elrdofs[1]] + Q[elrdofs[0]]) / 2.0

      uem = self.Eq.u(Qem)
      ulm = self.Eq.u(Qlm)
      urm = self.Eq.u(Qrm)
      umax = max(uem, ulm, urm)
      umin = min(uem, ulm, urm)

      uel = max(min(self.Eq.u(Q[eltdofs[0]]), umax), umin)
      uer = max(min(self.Eq.u(Q[eltdofs[1]]), umax), umin)

      uelr = self.Eq.u(np.array((Qlim[eltdofs[0],0], 2.0*Qem[1]-uer*Qlim[eltdofs[1],0])))
      uerl = self.Eq.u(np.array((Qlim[eltdofs[1],0], 2.0*Qem[1]-uel*Qlim[eltdofs[0],0])))

      if (abs(uer-uelr) < abs(uel-uerl)):
        dm = uer*Qlim[eltdofs[1],0] - Qem[1]
      else:
        dm = Qem[1] - uel*Qlim[eltdofs[0],0]

      corr = dm - dme
      if(np.min(self.Gr.elementneighbors[ielmt]) > -1):
        Qlim[eltdofs[0],1] = Qlim[eltdofs[0],1] - corr
        Qlim[eltdofs[1],1] = Qlim[eltdofs[1],1] + corr

      ## set momentum to zero if height is under wet tolerance
      #if np.max(Qlim[eltdofs,0]) < self.wettol:
        #mask = Qlim[eltdofs,0] < self.wettol
        #Qlim[eltdofs[mask],1] = 0.0

    # set momentum to zero if height is under wet tolerance
    mask = Qlim[:,0] < self.wettol
    Qlim[mask,1] = 0.0

    return Qlim
