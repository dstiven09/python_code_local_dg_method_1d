"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

functionality specific to elliptic equations, using the Local DG (LDG) method
"""

import numpy as np

# LDG coefficients for (zero) Dirichlet boundary conditions
c11d = 1.
c12d = 1.
c22d = 0.

# LDG coefficients for periodic boundary conditions
c11p = 1.
c12p = 1.
c22p = 0.

# LDG coefficients for reflection/wall boundary conditions
c11r = 1.
c12r = 1.
c22r = 1.


class LeftDirichlet:
  """
  left Dirchlet boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes Neumann boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0])
        ldat.append(c22d)
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0])
        ldat.append(-c22d)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes zero boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5-c12d)
        lcol.append(eltdofs[0] -1 + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5+c12d)

    return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes Neumann boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5-c12d)
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5+c12d)
      else:
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-1.)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes zero boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(c11d)
        lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        lrow.append(eltdofs[0]   + self.DGEl.doflength)
        ldat.append(-c11d)
      else:
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(c11d)

    return lrow, lcol, ldat

class LeftPeriodic:
  """
  left periodic boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0])
        ldat.append(+c22p)
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0])
        ldat.append(-c22p)
      else:
        lcol.append(self.DGEl.doflength-1)
        lrow.append(0)
        ldat.append(-c22p)
        lcol.append(0)
        lrow.append(0)
        ldat.append(+c22p)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5-c12p)
        lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5+c12p)
      else:
        lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        lrow.append(0)
        ldat.append(-0.5+c12p)
        lcol.append(0 + self.DGEl.doflength)
        lrow.append(0)
        ldat.append(-0.5-c12p)

    return lrow, lcol, ldat

  #def discr_numf1_var2_RBP_upwind(self):
    #"""
    #discretized numerical flux in first equation of second variable
    #imposes periodic boundary conditions

    #row of arr: determines the test function
    #column of arr: determines the dof of the unknown variable
    #"""

    #lrow, lcol, ldat = [], [], []
    #arr = np.zeros((self.DGEl.doflength,self.DGEl.doflength))
    #c12 = 0.

    ## element loop
    #for ielmt in range(self.Gr.elength):

      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        #lrow.append(eltdofs[0])
        #ldat.append(-1.+c12)
      #else:
        #lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        #lrow.append(0)
        #ldat.append(-1.+c12)

    #return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5-c12p)
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5+c12p)
      else:
        lcol.append(self.DGEl.doflength-1)
        lrow.append(0 + self.DGEl.doflength)
        ldat.append(-0.5-c12p)
        lcol.append(0)
        lrow.append(0 + self.DGEl.doflength)
        ldat.append(-0.5+c12p)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(+c11p)
        lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        lrow.append(eltdofs[0]   + self.DGEl.doflength)
        ldat.append(-c11p)
      else:
        lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        lrow.append(0 + self.DGEl.doflength)
        ldat.append(-c11p)
        lcol.append(0 + self.DGEl.doflength)
        lrow.append(0 + self.DGEl.doflength)
        ldat.append(+c11p)

    return lrow, lcol, ldat

class LeftReflection:
  """
  left reflection/wall boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []
    #arr = np.zeros((self.DGEl.doflength,self.DGEl.doflength))

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0])
        ldat.append(+c22r)
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0])
        ldat.append(-c22r)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5-c12r)
        lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-0.5+c12r)
      else:
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0])
        ldat.append(-1.)

    return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0]-1)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5-c12r)
        lcol.append(eltdofs[0])
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-0.5+c12r)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt>0):
        lcol.append(eltdofs[0] + self.DGEl.doflength)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(+c11r)
        lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        lrow.append(eltdofs[0] + self.DGEl.doflength)
        ldat.append(-c11r)

    return lrow, lcol, ldat

#class LeftReflectionnew:
  #"""
  #left reflection/wall (new try) boundary data for the elliptic problem, fluxes according to usual Poisson problem
  #row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  #column of arr: determines the dof of the unknown variable
  #"""

  #def __init__(self, Grid, DGElmt, Local):
    #self.Gr    = Grid
    #self.DGEl  = DGElmt
    #self.Local = Local

  #def bc11(self):
    #"""
    #discretized numerical flux in first equation of first variable
    #imposes Neumann boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c22 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0])
        #lrow.append(eltdofs[0])
        #ldat.append(c22)
        #lcol.append(eltdofs[0]-1)
        #lrow.append(eltdofs[0])
        #ldat.append(-c22)

    #return lrow, lcol, ldat

  #def bc12(self):
    #"""
    #discretized numerical flux in first equation of second variable
    #imposes zero boundary conditionsle
    #"""

    #lrow, lcol, ldat = [], [], []
    #c12 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0] + self.DGEl.doflength)
        #lrow.append(eltdofs[0])
        #ldat.append(-0.5-c12)
        #lcol.append(eltdofs[0] -1 + self.DGEl.doflength)
        #lrow.append(eltdofs[0])
        #ldat.append(-0.5+c12)
      #else:
        #lcol.append(eltdofs[0] + self.DGEl.doflength)
        #lrow.append(eltdofs[0])
        #ldat.append(-1.)

    #return lrow, lcol, ldat

  #def bc21(self):
    #"""
    #discretized numerical flux in second equation of first variable
    #imposes Neumann boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c12 = 1.
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0]-1)
        #lrow.append(eltdofs[0] + self.DGEl.doflength)
        #ldat.append(-0.5-c12)
        #lcol.append(eltdofs[0])
        #lrow.append(eltdofs[0] + self.DGEl.doflength)
        #ldat.append(-0.5+c12)
      #else:
        #lcol.append(eltdofs[0])
        #lrow.append(eltdofs[0] + self.DGEl.doflength)
        #ldat.append(1.)

    #return lrow, lcol, ldat

  #def bc22(self):
    #"""
    #discretized numerical flux in second equation of second variable
    #imposes zero boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c11 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0] + self.DGEl.doflength)
        #lrow.append(eltdofs[0] + self.DGEl.doflength)
        #ldat.append(c11)
        #lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        #lrow.append(eltdofs[0]   + self.DGEl.doflength)
        #ldat.append(-c11)

    #return lrow, lcol, ldat

class RightDirichlet:
  """
  left Dirchlet boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes Neumann boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1])
        ldat.append(c22d)
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1])
        ldat.append(-c22d)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes zero boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5-c12d)
        lcol.append(eltdofs[-1] +1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5+c12d)

    return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes Neumann boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5-c12d)
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5+c12d)
      else:
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(1.)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes zero boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(c11d)
        lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1]   + self.DGEl.doflength)
        ldat.append(-c11d)
      else:
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(c11d)

    return lrow, lcol, ldat

class RightPeriodic:
  """
  left periodic boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1])
        ldat.append(+c22p)
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1])
        ldat.append(-c22p)
      else:
        lcol.append(self.DGEl.doflength-1)
        lrow.append(self.DGEl.doflength-1)
        ldat.append(+c22p)
        lcol.append(0)
        lrow.append(self.DGEl.doflength-1)
        ldat.append(-c22p)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5-c12p)
        lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5+c12p)
      else:
        lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        lrow.append(self.DGEl.doflength-1)
        ldat.append(0.5-c12p)
        lcol.append(0 + self.DGEl.doflength)
        lrow.append(self.DGEl.doflength-1)
        ldat.append(0.5+c12p)

    return lrow, lcol, ldat

  #def discr_numf1_var2_RBP_upwind(self):
    #"""
    #discretized numerical flux in first equation of second variable
    #imposes periodic boundary conditions

    #row of arr: determines the test function
    #column of arr: determines the dof of the unknown variable
    #"""

    #lrow, lcol, ldat = [], [], []
    #arr = np.zeros((self.DGEl.doflength,self.DGEl.doflength))
    #c12 = 0.

    ## element loop
    #for ielmt in range(self.Gr.elength):

      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt>0):
        #lcol.append(eltdofs[0]-1 + self.DGEl.doflength)
        #lrow.append(eltdofs[0])
        #ldat.append(-1.+c12)
      #else:
        #lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        #lrow.append(0)
        #ldat.append(-1.+c12)

      #if (ielmt<self.Gr.elength-1):
        #lcol.append(eltdofs[-1] + self.DGEl.doflength)
        #lrow.append(eltdofs[-1])
        #ldat.append(1.-c12)
      #else:
        #lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        #lrow.append(self.DGEl.doflength-1)
        #ldat.append(1.-c12)

    #return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5-c12p)
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5+c12p)
      else:
        lcol.append(self.DGEl.doflength-1)
        lrow.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        ldat.append(+0.5+c12p)
        lcol.append(0)
        lrow.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        ldat.append(+0.5-c12p)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes periodic boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+c11p)
        lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1]   + self.DGEl.doflength)
        ldat.append(-c11p)
      else:
        lcol.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        lrow.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        ldat.append(+c11p)
        lcol.append(0 + self.DGEl.doflength)
        lrow.append(self.DGEl.doflength-1 + self.DGEl.doflength)
        ldat.append(-c11p)

    return lrow, lcol, ldat

class RightReflection:
  """
  left reflection/wall boundary data for the elliptic problem, fluxes according to usual Poisson problem
  row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  column of arr: determines the dof of the unknown variable
  """

  def __init__(self, Grid, DGElmt, Local):
    self.Gr    = Grid
    self.DGEl  = DGElmt
    self.Local = Local

  def bc11(self):
    """
    discretized numerical flux in first equation of first variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1])
        ldat.append(+c22r)
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1])
        ldat.append(-c22r)

    return lrow, lcol, ldat

  def bc12(self):
    """
    discretized numerical flux in first equation of second variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5-c12r)
        lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(0.5+c12r)
      else:
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1])
        ldat.append(1.)

    return lrow, lcol, ldat

  def bc21(self):
    """
    discretized numerical flux in second equation of first variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1]+1)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5-c12r)
        lcol.append(eltdofs[-1])
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+0.5+c12r)

    return lrow, lcol, ldat

  def bc22(self):
    """
    discretized numerical flux in second equation of second variable
    imposes reflecting boundary conditions
    """

    lrow, lcol, ldat = [], [], []

    # element loop
    for ielmt in range(self.Gr.elength):
      eltdofs = self.DGEl.elementdofs[ielmt]

      if (ielmt<self.Gr.elength-1):
        lcol.append(eltdofs[-1] + self.DGEl.doflength)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(+c11r)
        lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        lrow.append(eltdofs[-1] + self.DGEl.doflength)
        ldat.append(-c11r)

    return lrow, lcol, ldat

#class RightReflectionnew:
  #"""
  #left reflection/wall (new try) boundary data for the elliptic problem, fluxes according to usual Poisson problem
  #row of arr: determines the basis function, determing at which boundary (left or right) of the element we are
  #column of arr: determines the dof of the unknown variable
  #"""

  #def __init__(self, Grid, DGElmt, Local):
    #self.Gr    = Grid
    #self.DGEl  = DGElmt
    #self.Local = Local

  #def bc11(self):
    #"""
    #discretized numerical flux in first equation of first variable
    #imposes Neumann boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c22 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt<self.Gr.elength-1):
        #lcol.append(eltdofs[-1])
        #lrow.append(eltdofs[-1])
        #ldat.append(c22)
        #lcol.append(eltdofs[-1]+1)
        #lrow.append(eltdofs[-1])
        #ldat.append(-c22)

    #return lrow, lcol, ldat

  #def bc12(self):
    #"""
    #discretized numerical flux in first equation of second variable
    #imposes zero boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c12 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt<self.Gr.elength-1):
        #lcol.append(eltdofs[-1] + self.DGEl.doflength)
        #lrow.append(eltdofs[-1])
        #ldat.append(0.5-c12)
        #lcol.append(eltdofs[-1] +1 + self.DGEl.doflength)
        #lrow.append(eltdofs[-1])
        #ldat.append(0.5+c12)
      #else:
        #lcol.append(eltdofs[-1] + self.DGEl.doflength)
        #lrow.append(eltdofs[-1])
        #ldat.append(1.)

    #return lrow, lcol, ldat

  #def bc21(self):
    #"""
    #discretized numerical flux in second equation of first variable
    #imposes Neumann boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c12 = 1.
    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt<self.Gr.elength-1):
        #lcol.append(eltdofs[-1]+1)
        #lrow.append(eltdofs[-1] + self.DGEl.doflength)
        #ldat.append(+0.5-c12)
        #lcol.append(eltdofs[-1])
        #lrow.append(eltdofs[-1] + self.DGEl.doflength)
        #ldat.append(+0.5+c12)
      #else:
        #lcol.append(eltdofs[-1])
        #lrow.append(eltdofs[-1] + self.DGEl.doflength)
        #ldat.append(-1.)

    #return lrow, lcol, ldat

  #def bc22(self):
    #"""
    #discretized numerical flux in second equation of second variable
    #imposes zero boundary conditions
    #"""

    #lrow, lcol, ldat = [], [], []
    #c11 = 1.

    ## element loop
    #for ielmt in range(self.Gr.elength):
      #eltdofs = self.DGEl.elementdofs[ielmt]

      #if (ielmt<self.Gr.elength-1):
        #lcol.append(eltdofs[-1] + self.DGEl.doflength)
        #lrow.append(eltdofs[-1] + self.DGEl.doflength)
        #ldat.append(c11)
        #lcol.append(eltdofs[-1]+1 + self.DGEl.doflength)
        #lrow.append(eltdofs[-1]   + self.DGEl.doflength)
        #ldat.append(-c11)

    #return lrow, lcol, ldat
