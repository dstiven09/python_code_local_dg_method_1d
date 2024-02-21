"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

finite volume recovery
"""

import numpy as np

class PiecewConstRecovery:
  """
  Piecewise constant recovery for finite volumes

  Computes left and right interface values resulting from a piecewise constant
  reconstruction of cell mean values in each cell.
  """

  def __init__(self, Grid):
    """
    initializes reconstruction

    Input values:
      Grid    : grid structure
    """

    self.gr  = Grid

  def __call__(self, Q, t):
    """
    evaluates piecewise linear recovery

    Input values:
      Q       : vector of state variables
      t       : time level
      bvalue  : boundary values

    Return values:
      Ql : reconstructed left values
      Qr : reconstructed right values
    """

    Ql = np.copy(Q)
    Qr = np.copy(Q)

    return Ql, Qr


class PiecewLinRecovery(object):
  """
  Piecewise linear recovery for finite volumes

  Computes left and right interface values resulting from a piecewise linear
  reconstruction of cell mean values in each cell. The result strongly depends
  on the limiter which is used.
  """

  def __init__(self, Grid, Limiter, BValue):
    """
    initializes reconstruction

    Input values:
      Grid    : grid structure
      Limiter : limiter function for reconstruction
    """

    self.gr  = Grid
    self.Lim = Limiter
    self.bv  = BValue

  def __call__(self, Q, t):
    """
    evaluates piecewise linear recovery

    Input values:
      Q       : vector of state variables
      t       : time level
      BValue  : list of boundary values
      Grid    : grid structure
      limiter : limitation of reconstruction

    Return values:
      Ql : reconstructed left values
      Qr : reconstructed right values

    TODO: test if this works for other than uniform grids
    """

    if Q.ndim==1:
      nunknowns = 1
    else:
      nunknowns = Q.shape[1]

    DQ   = np.zeros((self.gr.elength, nunknowns))
    DQDx = np.zeros((self.gr.nlength, nunknowns))

    # inner edge loop for slope computations
    for inode in range(self.gr.inodes.shape[0]):

      ndelts = self.gr.inodeelements[inode]

      DQDx[self.gr.inodes[inode]] = 2.0 * (Q[ndelts[1]] - Q[ndelts[0]]) / \
        (self.gr.elementwidth[ndelts[1]] + self.gr.elementwidth[ndelts[0]])

    # boundary edge loops for slope computations
    for inode in range(self.gr.bnodes.shape[0]):

      ndelts = self.gr.bnodeelements[inode]

      if (self.bv[inode].side == 0):
        DQDx[self.gr.bnodes[inode]] = (Q[ndelts[1]] - self.bv[inode](Q, t)) / self.gr.elementwidth[ndelts[1]]
      else:
        DQDx[self.gr.bnodes[inode]] = (self.bv[inode](Q, t) - Q[ndelts[0]]) / self.gr.elementwidth[ndelts[0]]

    # element loop for recovery computations
    for ielmt in range(self.gr.elength):

      elmtnds   = self.gr.elementnodest[ielmt]
      DQ[ielmt] = self.Lim(DQDx[elmtnds[0]], DQDx[elmtnds[1]]) * self.gr.elementwidth[ielmt]

    Ql = Q - DQ / 2.0
    Qr = Q + DQ / 2.0

    return Ql, Qr


class PiecewLinRecoveryHydrostatic(PiecewLinRecovery):
  """
  PiecewLinRecoveryHydrostatic  hydrostatic second order recovery

  computes left and right interface values resulting from a second order
  hydrostatic reconstruction of cell mean values in each cell for the
  shallow water equations. The result strongly depends on the limiter which
  is used. The reconstruction is done as in Audusse et al. (2004).

  References:
  Audusse et al. (2004): "A Fast and Stable Well-Balanced Scheme with
  Hydrostatic Reconstruction for Shallow Water Flows", SIAM J. Sci. Comp.,
  Vol. 25(6), pp. 2050-2065
  """

  def __init__(self, Grid, Limiter, BValue, btopo):
    """
    initializes reconstruction

    Input values:
      Grid    : grid structure
      Limiter : limiter function for reconstruction
      btopo   : bottom topography
    """

    super(PiecewLinRecoveryHydrostatic, self).__init__(Grid, Limiter, BValue)
    self.b = btopo


  def __call__(self, Q, t):
    """
    Input values:
      Q       : vector of state variables
      t       : time level
      bvalue  : boundary values

    Return values:
      Ql : reconstructed left values
      Qr : reconstructed right values

    See also: PiecewConstRecoveryHydrostatic

    TODO % Check correctness of bvalue for reconstructed data
          (bbtopol/bbtopor)

    """

    Ql      = np.zeros((Q.shape))
    Qr      = np.zeros((Q.shape))
    QQ      = np.zeros((self.gr.elength, 3))
    QQ[:,0] = Q[:,0]
    nonzeroh = Q[:,0] > 1e-8
    QQ[nonzeroh,1] = Q[nonzeroh,1] / Q[nonzeroh,0]
    QQ[:,2] = Q[:,0]+self.b

    # reconstruct in h, u, and h+b and compute bl, br from this
    [QQl, QQr] = super(PiecewLinRecoveryHydrostatic, self).__call__(QQ, t)
    bl = QQl[:,2] - QQl[:,0]
    br = QQr[:,2] - QQr[:,0]

    # inner edge loop
    for inode in range(self.gr.inodes.shape[0]):

      ndelts = self.gr.inodeelements[inode]

      bh = np.maximum(br[ndelts[0]], bl[ndelts[1]])
      hm = np.maximum(QQr[ndelts[0],2] - bh, 0.0)
      hp = np.maximum(QQl[ndelts[1],2] - bh, 0.0)

      Qr[ndelts[0],0] = hm
      Qr[ndelts[0],1] = hm * QQr[ndelts[0],1]
      Ql[ndelts[1],0] = hp
      Ql[ndelts[1],1] = hp * QQl[ndelts[1],1]

    # boundary edge loop
    for inode in range(self.gr.bnodes.shape[0]):

      ndelts = self.gr.bnodeelements[inode]

      if (self.bv[inode].side == 0):
        bh = bl[ndelts[1]]
        hp = np.maximum(QQl[ndelts[1],2] - bh, 0.0)

        Ql[ndelts[1],0] = hp
        Ql[ndelts[1],1] = hp * QQl[ndelts[1],1]

      else:
        bh = br[ndelts[0]]
        hm = np.maximum(QQr[ndelts[0],2] - bh, 0.0)

        Qr[ndelts[0],0] = hm
        Qr[ndelts[0],1] = hm * QQr[ndelts[0],1]

    return Ql, Qr
