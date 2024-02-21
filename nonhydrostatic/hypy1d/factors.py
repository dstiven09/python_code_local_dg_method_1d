"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2016)

functionality specific to elliptic equations, using the Local DG method (LDG)
"""

import numpy as np
import scipy.sparse.linalg as linalg
import scipy.sparse as sp
import time

class FactorsElliptic:
  """
  discretize factors for each term in the LDG equation system of the elliptic equation,
  according to the input defined in the class Eqell in equation.py
  """

  def __init__(self, Grid, Source, Equell, DGElmt, Quad):
    self.Gr    = Grid
    self.Src   = Source
    self.Eqell = Equell
    self.DGEl  = DGElmt
    self.Quad  = Quad

  def discr_f1_var2(self, Q):
    """
    discretized factor f1_var2 including gradient
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      M       = np.zeros((len(eltdofs), len(eltdofs)))
      bi      = self.Eqell.btopo[eltdofs]
      F       = self.Eqell.fact_f1_var2(Q[eltdofs])

      for idof in range(len(eltdofs)):
        M[:,idof] = -self.elmtfact(t, self.Quad.dpsidxi)[:,idof] * F[idof]
        for i in range(len(eltdofs)):
          lcol.append(idof + eltdofs[0] + self.DGEl.doflength)
          lrow.append(i + eltdofs[0])
          ldat.append(M[i,idof])

    return lrow, lcol, ldat

  def discr_s1_var1(self, Q, dt):
    """
    discretized factor s1_var1
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      bi      = self.Eqell.btopo[eltdofs]
      M       = np.zeros((len(eltdofs), len(eltdofs)))
      F       = self.Eqell.fact_s1_var1(Q[eltdofs], bi, dt)

      for idof in range(len(eltdofs)):
        M[:,idof] = self.elmtfact(t, self.Quad.psi)[:,idof] * F[idof] *self.DGEl.J[ielmt]
        for i in range(len(eltdofs)):
          lcol.append(idof + eltdofs[0])
          lrow.append(i + eltdofs[0])
          ldat.append(M[i,idof])

    return lrow, lcol, ldat

  def discr_s1_var2(self, Q):
    """
    discretized factor s1_var2 including gradient
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      M = self.elmtfact_3_s1(ielmt, Q[eltdofs], t)*self.DGEl.J[ielmt]
      for i in range(len(eltdofs)):
        for j in range(len(eltdofs)):
          lcol.append(j + eltdofs[0] + self.DGEl.doflength)
          lrow.append(i + eltdofs[0])
          ldat.append(M[i,j])

    return lrow, lcol, ldat

  def elmtfact_3_s1(self, ielmt, Qelmt, t):
    """
    compute source term as a factor for one element
    """

    Msrc = np.zeros((self.Quad.quadpts, self.DGEl.edofs))
    Mnew = np.zeros((self.DGEl.edofs, self.Quad.quadpts))

    dhi  = np.dot((self.DGEl.ddx[ielmt]).T, Qelmt[:,0])
    dbi  = np.dot((self.DGEl.ddx[ielmt]).T, self.Eqell.btopo[self.DGEl.elementdofs[ielmt]])
    bi   = self.Eqell.btopo[self.DGEl.elementdofs[ielmt]]
    Fact = self.Eqell.fact_s1_var2(Qelmt, dhi, dbi, bi)
    Mdb  = np.dot(self.Quad.psi.T, Fact)

    for iquad in range(self.Quad.quadpts):
      for idof in range(self.DGEl.edofs):
        Msrc[iquad,idof] = Mdb[iquad] * self.Quad.psi.T[iquad,idof]

    for iquad in range(self.Quad.quadpts):
      Mnew[:,iquad] = self.Quad.eMinvpsi[:,iquad] * self.Quad.w[iquad]

    return np.dot(Mnew,Msrc)

  def discr_s1_c(self, Q, dt):
    """
    discretized factor s1_c including gradient
    """

    arr = np.zeros(self.DGEl.doflength)
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      bi      = self.Eqell.btopo[eltdofs]
      rhs     = self.Eqell.fact_s1_c(Q[eltdofs], bi, dt)

      arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi), rhs)*self.DGEl.J[ielmt]

    return arr

  def discr_s1_c_add(self, Q, Qold):
    """
    discretized factor s1_c including gradient
    """

    arr = np.zeros(self.DGEl.doflength)
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      dqiold  = np.dot((self.DGEl.ddx[ielmt]).T, Qold[eltdofs])
      bi      = self.Eqell.btopo[eltdofs]
      dbi     = np.dot((self.DGEl.ddx[ielmt]).T, bi)
      rhs     = self.Eqell.fact_s1_c_add(Q[eltdofs], Qold[eltdofs], dqiold, bi, dbi)

      arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi), rhs)*self.DGEl.J[ielmt]

    return arr

  def discr_f2_var1(self, Q):
    """
    discretized factor f2_var1 including gradient
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      M       = np.zeros((len(eltdofs), len(eltdofs)))
      F       = self.Eqell.fact_f2_var1(Q[eltdofs])

      for idof in range(len(eltdofs)):
        M[:,idof] = -self.elmtfact(t, self.Quad.dpsidxi)[:,idof] * F[idof]
        for i in range(len(eltdofs)):
          lcol.append(idof + eltdofs[0])
          lrow.append(i + eltdofs[0] + self.DGEl.doflength)
          ldat.append(M[i,idof])

    return lrow, lcol, ldat

  def discr_s2_var1(self, Q, dt):
    """
    discretized factor s2_var1
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      M = self.elmtfact_3_s2(ielmt, Q[eltdofs], dt, t)*self.DGEl.J[ielmt]
      for i in range(len(eltdofs)):
        for j in range(len(eltdofs)):
          lcol.append(j + eltdofs[0])
          lrow.append(i + eltdofs[0] + self.DGEl.doflength)
          ldat.append(M[i,j])

    return lrow, lcol, ldat

  def elmtfact_3_s2(self, ielmt, Qelmt, dt, t):
    """
    compute source term as a matrix factor in one element
    """

    Msrc = np.zeros((self.Quad.quadpts, self.DGEl.edofs))
    Mnew = np.zeros((self.DGEl.edofs, self.Quad.quadpts))

    dhi  = np.dot((self.DGEl.ddx[ielmt]).T, Qelmt[:,0])
    bi   = self.Eqell.btopo[self.DGEl.elementdofs[ielmt]]
    dbi  = np.dot((self.DGEl.ddx[ielmt]).T, bi)
    Fact = self.Eqell.fact_s2_var1(Qelmt, dhi, bi, dbi, dt)
    Mdb  = np.dot(self.Quad.psi.T, Fact)

    for iquad in range(self.Quad.quadpts):
      for idof in range(self.DGEl.edofs):
        Msrc[iquad,idof] = Mdb[iquad] * self.Quad.psi.T[iquad,idof]

    for iquad in range(self.Quad.quadpts):
      Mnew[:,iquad] = self.Quad.psi[:,iquad] * self.Quad.w[iquad]

    return np.dot(Mnew,Msrc)

  def discr_s2_var2(self, Q, dt):
    """
    discretized factor s2_var2
    """

    lrow, lcol, ldat = [], [], []
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      bi      = self.Eqell.btopo[eltdofs]
      M       = np.zeros((len(eltdofs), len(eltdofs)))
      F       = self.Eqell.fact_s2_var2(Q[eltdofs], bi, dt)

      for idof in range(len(eltdofs)):
        M[:,idof] = self.elmtfact(t, self.Quad.psi)[:,idof] * F[idof] * self.DGEl.J[ielmt]
        for i in range(len(eltdofs)):
          lcol.append(idof + eltdofs[0] + self.DGEl.doflength)
          lrow.append(i + eltdofs[0] + self.DGEl.doflength)
          ldat.append(M[i,idof])

    return lrow, lcol, ldat

  def discr_s2_c(self, Q):
    """
    discretized factor s2_c
    """

    lrow, lcol, ldat = [], [], []
    arr = np.zeros(self.DGEl.doflength)
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      bi      = self.Eqell.btopo[eltdofs]
      rhs     = self.Eqell.fact_s2_c(Q[eltdofs], bi, ielmt)

      arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi), rhs)*self.DGEl.J[ielmt]

    return arr

  def discr_s2_c_add(self, Q, Qold, dt):
    """
    discretized factor s1_c including gradient
    """

    arr = np.zeros(self.DGEl.doflength)
    t = 0.

    # element loop
    for ielmt in range(self.Gr.elength):

      eltdofs = self.DGEl.elementdofs[ielmt]
      bi      = self.Eqell.btopo[eltdofs]
      rhs     = self.Eqell.fact_s2_c_add(Q[eltdofs], Qold[eltdofs], bi, dt)

      arr[eltdofs] = np.dot(self.elmtfact(t, self.Quad.psi), rhs)*self.DGEl.J[ielmt]

    return arr

  def elmtfact(self, t, M):
    """
    compute discretized factor for one element
    """

    Mnew = np.zeros(M.shape)

    # compute flux divergence at quadrature points
    for iquad in range(self.Quad.quadpts):
      Mnew[:,iquad] = M[:,iquad] * self.Quad.w[iquad]

    return np.dot(Mnew,self.Quad.psi.T)

class Solve:
  """
  solve the elliptic equation
  """

  def Solve_sep(self, RHS, Q):
    """
    for debugging: solve each of both equations seperately
    """

    Q = RHS(Q, 0)

    return Q

  def Solve_linalg(self, A, b, t):
    """
    solve equation system with numpy routine linalg.solve
    """
    x = np.linalg.solve(A.toarray(), b)
    if (np.allclose(np.dot(A.toarray(), x), b)==True):
      print('System solved successfully')
    else:
      print('Attention: System not solved properly!')
    rank = np.linalg.matrix_rank(A.toarray())
    if (rank<len(b)):
      print('Attention: Matrix has no full rank! ', 'dofs-rank= ', len(b)-rank)

    if (t==0.):
      filestr = 'linalg_cond.out'
      f = open(filestr, 'a')
      f.write('factors_4: m=' + str(len(b)/4 + 1) + ', condition number: ' + str(np.linalg.cond(A.toarray())) + '\n')
      f.close()

    return x

  def Solve_gmres(self, A, b, Q, dofs1, dofs2, t):
    """
    solve equation system with lgmres algorithm
    """

    # guess for x: take solution of last timestep
    doflen = len(dofs1)
    xtry = np.zeros(2*doflen)
    xtry[dofs1] = Q[:,1]
    xtry[dofs2] = Q[:,3]

    filestr = 'test_tol_lgmres.out'
    if (t==0.):
      f = open(filestr, 'a')

    x, i = linalg.lgmres(A, b, xtry, 1e-12)
    if (i==0):
      print('System solved successfully')
    elif (i>0):
      print('convergence to tolerance not achieved, number of iterations: ', i)
      f = open(filestr, 'a')
      f.write(str(doflen/2+1) + ", " + str(t) + '\n')
      f.close()
    else:
      print('illegal input or breakdown')

    return x

  def Solve_schur(self, A1, b, Q, dofs1, dofs2):
    """
    solve equation system with Schur complement
    """

    # decomposition of matrix A1 into its 4 block matrizes:
    doflen = len(dofs1)
    x = np.zeros(2*doflen)
    A = (A1.toarray())[0:doflen, 0:doflen]
    B = A1.toarray()[0:doflen, doflen:2*doflen]
    C = A1.toarray()[doflen:2*doflen, 0:doflen]
    D = A1.toarray()[doflen:2*doflen, doflen:2*doflen]

    CAm1 = np.dot(C,np.linalg.inv(A))
    S2   = D - np.dot(CAm1, B)
    b2   = b[dofs2] - np.dot(CAm1, b[dofs1])

    x[dofs2], i1 = linalg.lgmres(S2, b2, np.zeros(doflen), 1e-12)
    if (i1==0):
      print('System solved successfully: dofs2')
    elif (i1>0):
      print('convergence to tolerance not achieved, number of iterations: ', i1)
    else:
      print('illegal input or breakdown')


    b1   = b[dofs1] - np.dot(B, x[dofs2])
    x[dofs1], i2 = linalg.lgmres(A, b1, np.zeros(doflen), 1e-12)
    if (i2==0):
      print('System solved successfully: dofs1')
    elif (i2>0):
      print('convergence to tolerance not achieved, number of iterations: ', i1)
    else:
      print('illegal input or breakdown')

    return x

  def coo_matrix_arrays(self, fact, Q, dt, Bvell):
    """
    construct sparse matrix in format coo_matrix
    """

    # inner components
    lr,  lc,  ld  = fact.discr_s1_var1(Q, dt)
    lr1, lc1, ld1 = fact.discr_f1_var2(Q)
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = fact.discr_s1_var2(Q)
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = fact.discr_f2_var1(Q)
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = fact.discr_s2_var1(Q, dt)
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = fact.discr_s2_var2(Q, dt)
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))


    # boundary conditions
    lr1, lc1, ld1 = Bvell[0].bc11()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[0].bc12()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[0].bc21()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[0].bc22()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[1].bc11()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[1].bc12()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[1].bc21()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    lr1, lc1, ld1 = Bvell[1].bc22()
    lr = np.concatenate((lr, lr1))
    lc = np.concatenate((lc, lc1))
    ld = np.concatenate((ld, ld1))

    return lr, lc, ld

  def Solve_sys(self, fact, Q, dt, t, Bvell, Qold):
    """
    construct and solve the linear equation system for the non-hydrostatic pressure (second unknown) and the updated horizontal velocity hu (first unknown)
    """

    # set boundary conditions
    doflen   = fact.DGEl.doflength
    bc_1     = np.zeros(doflen)
    bc_2     = np.zeros(doflen)
    if (len(Bvell[2])==2):  # non-zero Dirichlet
      F        = np.zeros(doflen)
      F[0]     = Bvell[2][0]
      F[-1]    = Bvell[2][1]
      lr, lc, ld = fact.discr_f1_var2(Q)
      B        = sp.coo_matrix( (ld,(lr,lc)), shape=(doflen*2,doflen*2)).toarray()
      bc_1     = -np.dot(B[0:doflen, doflen:doflen*2], F)
      bc_2[0]  = Bvell[2][0]
      bc_2[-1] = -Bvell[2][1]

    # adjustments for 2nd order timestepping
    if (fact.Eqell.nht==2):
      rhs1_2nd = fact.discr_s1_c_add(Q, Qold)
      rhs2_2nd = fact.discr_s2_c_add(Q, Qold, dt)
    else:
      rhs1_2nd = np.zeros(doflen)
      rhs2_2nd = np.zeros(doflen)

    # set up right-hand side
    b     = np.zeros(2*doflen)
    dofs1 = np.arange(0,doflen,1)
    dofs2 = np.arange(doflen,doflen*2,1)
    b[dofs1] = fact.discr_s1_c(Q, dt) + bc_1 + bc_2 + rhs1_2nd
    b[dofs2] = fact.discr_s2_c(Q) + rhs2_2nd

    # set up and solve linear equation system
    row, col, data = self.coo_matrix_arrays(fact, Q, dt, Bvell)
    A = sp.coo_matrix( (data,(row,col)), shape=(doflen*2,doflen*2))
    #print A.toarray()
    x = self.Solve_gmres(A, b, Q, dofs1, dofs2, t)
    #x = self.Solve_schur(A, b, Q, dofs1, dofs2)
    #x = self.Solve_linalg(A, b, t)

    #update non-zero Dirichlet boundary conditions to solution of zero boundary problem
    if (len(Bvell[2])==2):
      x[len(dofs1)+dofs1[0]] = x[len(dofs1)+dofs1[0]] + Bvell[2][0]
      x[len(dofs1)+dofs1[-1]] = x[len(dofs1)+dofs1[-1]] + Bvell[2][1]

    return x[dofs1], x[dofs2]

  def corr_tracer(self, fact, FD, Qnew, hu, pnh, t, dt, Qold):
    """
    correction step: update h and hw, copy hu
    """

    # initializations
    Qnew2 = np.zeros(Qnew.shape)
    hw    = np.zeros(Qnew[:,2].shape)

    # in case of second order timestepping: h-corrector
    if (fact.Eqell.nht==2):
      h     = np.zeros(fact.DGEl.doflength)
      start = np.zeros((fact.DGEl.doflength, FD.Eq.unknowns))
      Qhelp = np.zeros((fact.DGEl.doflength, FD.Eq.unknowns))
      Qhelp[:,0] = Qnew[:,0]
      Qhelp[:,1] = hu - Qnew[:,1]

      # element loop:
      for ielmt in range(FD.Gr.elength):
        eltdofs = fact.DGEl.elementdofs[ielmt]
        flxh   = fact.Eqell.corr_hflx(Qnew[eltdofs], hu[eltdofs], dt)
        h[eltdofs]  = Qnew[eltdofs,0] + np.dot(fact.elmtfact(t, fact.Quad.eMinvdpsidxi), flxh) /fact.DGEl.J[ielmt]

      edgeflx = FD.edge_computations(start, Qhelp, t)
      h       = h + dt*edgeflx[:,0]
      Qnew2[:,0] = h
    if (fact.Eqell.nht<=1):
      Qnew2[:,0] = Qnew[:,0]
    #Qnew2[:,0] = Qnew[:,0]

    # hw-corrector using corrected h (which is already corrected in case of nht=2)
    for ielmt in range(FD.Gr.elength):

      eltdofs = fact.DGEl.elementdofs[ielmt]
      rhs_hw = fact.Eqell.corr_hw(Qnew[eltdofs], pnh[eltdofs], dt, Qold[eltdofs])
      hw[eltdofs] = np.dot(fact.elmtfact(t, FD.Quad.eMinvpsi), rhs_hw)

    Qnew2[:,1] = hu
    Qnew2[:,2] = hw
    Qnew2[:,3] = pnh

    return Qnew2

  #def corr_tracer_new(self, fact, FD, Qnew, hu, pnh, t, dt, Qhelp, Qold):
    #"""
    #correction step: update h and hw, copy hu
    #"""

    ## initializations
    #Qnew2 = np.zeros(Qnew.shape)
    #Qnewh = np.copy(Qnew)
    #Qnewh[:,1] = hu
    #hw    = np.zeros(Qnew[:,2].shape)

    ## in case of second order timestepping: h-corrector
    #if (fact.Eqell.nht==2):
      #flux1 = self.fluxmy(fact, FD, Qold, dt, t)
      #flux2 = self.fluxmy(fact, FD, Qnewh, dt, t)
      #Qnew2[:,0] = Qold[:,0] - dt/2.*(flux1 + flux2)
    #if (fact.Eqell.nht<=1):
      #Qnew2[:,0] = Qnew[:,0]
    ##Qnew2[:,0] = Qnew[:,0]

    ## hw-corrector using non-corrected h
    #for ielmt in range(FD.Gr.elength):

      #eltdofs = fact.DGEl.elementdofs[ielmt]
      #bi     = fact.Eqell.btopo[eltdofs]
      #dbi    = np.dot((fact.DGEl.ddx[ielmt]).T, bi)
      #ddbi   = np.dot((fact.DGEl.ddxx[ielmt]).T, bi)
      #dxi    = np.dot((fact.DGEl.ddx[ielmt]).T, Qnew[eltdofs,0]+bi-fact.Eqell.d)
      #dxiold = np.dot((fact.DGEl.ddx[ielmt]).T, Qhelp[eltdofs,0]+bi-fact.Eqell.d)
      #rhs_hw = fact.Eqell.corr_hw(Qnew[eltdofs], (Qnew[:,2])[eltdofs], bi, pnh[eltdofs], dbi, ddbi, dxi, dt, Qhelp[eltdofs], dxiold)
      #hw[eltdofs] = np.dot(fact.elmtfact(t, FD.Quad.eMinvpsi), rhs_hw)

    #Qnew2[:,1] = hu
    #Qnew2[:,2] = hw
    #Qnew2[:,3] = pnh

    #return Qnew2

  #def fluxmy(self, fact, FD, Q, dt, t):
    #"""
    #compute Q_x
    #"""

    #flux  = np.zeros(Q[:,0].shape)
    #start = np.zeros((fact.DGEl.doflength, FD.Eq.unknowns))

    ## element loop:
    #for ielmt in range(FD.Gr.elength):
      #eltdofs = fact.DGEl.elementdofs[ielmt]
      #flux[eltdofs]  = np.dot(fact.elmtfact(t, fact.Quad.eMinvdpsidxi), Q[eltdofs])/fact.DGEl.J[ielmt]

    #edgeflx = FD.edge_computations(start, Q, t)

    #flux  = flux + edgeflx[:,0]

    #return flux
