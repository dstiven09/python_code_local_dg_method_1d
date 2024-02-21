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
    #phitmp = np.zeros(self.Gr.elength)

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      elldofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,0]]
      elrdofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,1]]

      Helt  = H[eltdofs]
      belt  = self.Src.btopo[eltdofs]
      iHmin = np.argmin(Helt)
      #iswetelt = (Helt[iHmin]-np.max(belt) >= self.wettol)
      iswetelt = (Helt[iHmin]-np.max(belt) >= 0.0)

      # limit height, differentiate between wet and (semi) dry elements
      #if (iswetelt):

      # for wet elements limit in hydrostatic variable (h+b)
      dHe = (H[eltdofs[1]] - H[eltdofs[0]]) / 2.0
      Hem = (H[eltdofs[1]] + H[eltdofs[0]]) / 2.0
      Hlm = (H[elldofs[1]] + H[elldofs[0]]) / 2.0
      Hrm = (H[elrdofs[1]] + H[elrdofs[0]]) / 2.0

      dHmax = np.maximum(np.maximum(Hlm, Hem), Hrm) - Hem
      dHmin = Hem - np.minimum(np.minimum(Hlm, Hem), Hrm)
      dH    = np.sign(dHe) * np.minimum(abs(dHe), np.minimum(dHmax, dHmin))
      corrH  = dH - dHe
      QlimH = Qlim[eltdofs,0]
      if(np.min(self.Gr.elementneighbors[ielmt]) > -1):
        QlimH[0] = Qlim[eltdofs[0],0] - corrH
        QlimH[1] = Qlim[eltdofs[1],0] + corrH

      #else:

      # for (semi) dry elements limit in primitive variable (h)
      dhe = (Q[eltdofs[1],0] - Q[eltdofs[0],0]) / 2.0
      hem = (Q[eltdofs[1],0] + Q[eltdofs[0],0]) / 2.0
      hlm = (Q[elldofs[1],0] + Q[elldofs[0],0]) / 2.0
      hrm = (Q[elrdofs[1],0] + Q[elrdofs[0],0]) / 2.0

      dhmax = np.maximum(np.maximum(hlm, hem), hrm) - hem
      dhmin = hem - np.minimum(np.minimum(hlm, hem), hrm)
      dh    = np.sign(dhe) * np.minimum(abs(dhe), np.minimum(dhmax, dhmin))
      corrh  = dh - dhe
      Qlimh = Qlim[eltdofs,0]
      if(np.min(self.Gr.elementneighbors[ielmt]) > -1):
        Qlimh[0] = Qlim[eltdofs[0],0] - corrh
        Qlimh[1] = Qlim[eltdofs[1],0] + corrh

      if (np.max(belt)-np.min(belt)) > self.wettol:
        phi = np.maximum(0.0, np.minimum(1.0, (np.min(Helt)-np.min(belt))/(np.max(belt)-np.min(belt))))
      else:
        phi = 1.0
      #phitmp[ielmt] = phi

      #Qlim[eltdofs,0] = phi * QlimH + (1.0-phi) * Qlimh
      corrphi = phi * corrH + (1.0-phi) * corrh
      if(np.min(self.Gr.elementneighbors[ielmt]) > -1):
        Qlim[eltdofs[0],0] = Qlim[eltdofs[0],0] - corrphi
        Qlim[eltdofs[1],0] = Qlim[eltdofs[1],0] + corrphi

      helt  = Qlim[eltdofs,0]
      ihmin = np.argmin(helt)
      if (helt[ihmin] < 0.0):
        helt[1-ihmin]   = helt[1-ihmin] + helt[ihmin]
        helt[ihmin]     = 0.0
        Qlim[eltdofs,0] = helt

      # try to correct some round off errors
      if (Qlim[eltdofs[0],0] < self.fin.eps):
        Qlim[eltdofs[0],0] = 0.0
      if (Qlim[eltdofs[1],0] < self.fin.eps):
        Qlim[eltdofs[1],0] = 0.0

      # check for positivity of mean height
      if (Qlim[eltdofs[0],0]+Qlim[eltdofs[1],0] < 0.0):
        print(Qlim[eltdofs,0])
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

    # set momentum to zero if height is under wet tolerance
    mask = Qlim[:,0] < self.wettol
    Qlim[mask,1] = 0.0

    #print(phitmp)

    return Qlim
