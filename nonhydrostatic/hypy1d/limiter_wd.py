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
    tbw
    """

    tol = 1.0e-8

    Qlim   = np.copy(Q)
    Qhy    = np.copy(Q)
    Qhy[:,0] = Qhy[:,0] + self.Src.btopo

    # element loop for inner element computations
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      elldofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,0]]
      elrdofs = self.DGEl.elementdofs[self.Gr.elementneighbors[ielmt,1]]

      Qelt  = Q[elldofs]
      belt  = self.Src.btopo[elldofs]
      xhmax = np.argmax(Qelt[:,0])
      xhmin = np.argmin(Qelt[:,0])
      #iswetell = (Qelt[xhmin,0] >= tol or Qelt[xhmax,0]+belt[xhmax]-np.max(belt) >= tol)
      iswetell = (Qelt[xhmin,0] >= tol)

      Qelt  = Q[elrdofs]
      belt  = self.Src.btopo[elrdofs]
      xhmax = np.argmax(Qelt[:,0])
      xhmin = np.argmin(Qelt[:,0])
      #iswetelr = (Qelt[xhmin,0] >= tol or Qelt[xhmax,0]+belt[xhmax]-np.max(belt) >= tol)
      iswetelr = (Qelt[xhmin,0] >= tol)

      Qelt  = Q[eltdofs]
      belt  = self.Src.btopo[eltdofs]
      xhmax = np.argmax(Qelt[:,0])
      xhmin = np.argmin(Qelt[:,0]+belt)
      #iswetelt = (Qelt[xhmin,0] >= tol or Qelt[xhmax,0]+belt[xhmax]-np.max(belt) >= tol)
      iswetelt = (Qelt[xhmin,0]+belt[xhmin]-np.max(belt) >= tol)
      #if (iswetelt and iswetell and iswetelr):
      if 0:#(iswetelt):

        dQe = (Qhy[eltdofs[1]] - Qhy[eltdofs[0]])
        Qem = (Qhy[eltdofs[1]] + Qhy[eltdofs[0]]) / 2.0
        Qlm = (Qhy[elldofs[1]] + Qhy[elldofs[0]]) / 2.0
        Qrm = (Qhy[elrdofs[1]] + Qhy[elrdofs[0]]) / 2.0
        dQl = Qem - Qlm
        dQr = Qrm - Qem

        dQ = np.zeros(dQe.shape)
        if (np.sign(dQe[0]) == np.sign(dQl[0]) and np.sign(dQe[0]) == np.sign(dQr[0])):
          dQ[0] = np.sign(dQe[0]) * min(abs(dQe[0]), abs(dQl[0]), abs(dQr[0]))
          #dQ[0] = np.sign(dQe[0]) * min(abs(dQe[0]), abs((dQl[0]+dQr[0])/2.0), abs(2.0*dQl[0]), abs(2.0*dQr[0]))

        if (np.sign(dQe[1]) == np.sign(dQl[1]) and np.sign(dQe[1]) == np.sign(dQr[1])):
          dQ[1] = np.sign(dQe[1]) * min(abs(dQe[1]), abs(dQl[1]), abs(dQr[1]))
          #dQ[1] = np.sign(dQe[1]) * min(abs(dQe[1]), abs((dQl[1]+dQr[1])/2.0), abs(2.0*dQl[1]), abs(2.0*dQr[1]))

        corr = (dQ - dQe) / 2.0
        Qlim[eltdofs[0]] = Qlim[eltdofs[0]] - corr
        Qlim[eltdofs[1]] = Qlim[eltdofs[1]] + corr

      else:

        dQe = (Q[eltdofs[1]] - Q[eltdofs[0]])
        Qem = (Q[eltdofs[1]] + Q[eltdofs[0]]) / 2.0
        Qlm = (Q[elldofs[1]] + Q[elldofs[0]]) / 2.0
        Qrm = (Q[elrdofs[1]] + Q[elrdofs[0]]) / 2.0
        dQl = Qem - Qlm
        dQr = Qrm - Qem

        dQ = np.zeros(dQe.shape)
        if (np.sign(dQe[0]) == np.sign(dQl[0]) and np.sign(dQe[0]) == np.sign(dQr[0])):
          dQ[0] = np.sign(dQe[0]) * min(abs(dQe[0]), abs(dQl[0]), abs(dQr[0]))
          #dQ[0] = np.sign(dQe[0]) * min(abs(dQe[0]), abs((dQl[0]+dQr[0])/2.0), abs(2.0*dQl[0]), abs(2.0*dQr[0]))

        if (np.sign(dQe[1]) == np.sign(dQl[1]) and np.sign(dQe[1]) == np.sign(dQr[1])):
          dQ[1] = np.sign(dQe[1]) * min(abs(dQe[1]), abs(dQl[1]), abs(dQr[1]))
          #dQ[1] = np.sign(dQe[1]) * min(abs(dQe[1]), abs((dQl[1]+dQr[1])/2.0), abs(2.0*dQl[1]), abs(2.0*dQr[1]))

        corr = (dQ - dQe) / 2.0
        Qlim[eltdofs[0]] = Qlim[eltdofs[0]] - corr
        Qlim[eltdofs[1]] = Qlim[eltdofs[1]] + corr

      Qelt  = Qlim[eltdofs]
      xhmax = np.argmax(Qelt[:,0])
      xhmin = np.argmin(Qelt[:,0])
      if (Qelt[xhmin,0] < 0.0):
        print('Elt. with neg. values: '+str(ielmt))
        #Qelt[xhmax,0] = Qelt[xhmax,0] + Qelt[xhmin,0]
        Qelt[xhmax] = Qelt[xhmax] + Qelt[xhmin]
        Qelt[xhmin]   = 0.0
        Qlim[eltdofs] = Qelt

      #xhmax = np.argmax(Qelt[:,0])
      #if (Qelt[xhmax,0] < tol):
        #Qlim[eltdofs,1] = 0.0


    mask = Qlim[:,0] < tol
    Qlim[mask,1] = 0.0

    if (np.min(Qlim[:,0]) < -1.0e-17):
      mask = Qlim[:,0] < -1.0e-17
      print(np.where(mask))
      print(Qlim[mask,0])
    mask = Qlim[:,0] < 1.0e-17
    Qlim[mask,0] = 0.0

    return Qlim
