"""
evaluation of goal functionals
Susanne Beckers (2015)

"""

import numpy as np

def PotEnergy(Q, DGElmt, Quad):
      
  goal = 0.0
  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]

    Qq     = np.dot(Quad.psi.T, Q[eltdofs])
    IntQ   = DGElmt.J[ielt] * np.dot(Quad.w, Qq)
    goal   = goal + IntQ

  return goal
  
    

def MaxHeight(Q, m):
      
  goal = np.zeros(1)
  goal = max(Q[:])
    
  return goal
  
def L1error(Q, Gr, DGElmt, Quad, f):
  """
  compute L_1 error of a DG discretization with respect to a given function f
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    deltax  = eltcoo[1]-eltcoo[0]
    xq = (Quad.x+1.0)/2.0 * deltax + eltcoo[0]
    F  = np.zeros(xq.shape[0])
    for i in range(xq.shape[0]):
         F[i] = f(xq[i])
    QHelp  = np.dot(Quad.psi.T, Q[eltdofs])
    Qqsq   = abs(QHelp[:,0] - F)
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return IQsq
  

def L2error(Q, Gr, DGElmt, Quad, f):
  """
  compute L_2 error of a DG discretization with respect to a given function f
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    deltax  = eltcoo[1]-eltcoo[0]
    xq = (Quad.x+1.0)/2.0 * deltax + eltcoo[0]
    F  = np.zeros(xq.shape[0])
    for i in range(xq.shape[0]):
         F[i] = f(xq[i])
    QHelp  = np.dot(Quad.psi.T, Q[eltdofs])
    Qqsq   = (QHelp[:,0] - F)**2
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return np.sqrt(IQsq)
  

def H1error(Q, Gr, DGElmt, Quad, Quad2, f, df):
  """
  compute H_1 error of a DG discretization with respect to a given function f
  """
  # compute dQ element wise
  dQ  = np.zeros(Q.shape)
  for ielmt in range(Gr.elength):
      eltdofs = DGElmt.elementdofs[ielmt]
      Qelmt   = Q[eltdofs,:]
      ddxelt  = DGElmt.ddx[ielmt]
     # compute derivatives at quadrature points
      ddxq    = np.dot(ddxelt, Quad.psi)
      dQ[eltdofs,0] = np.dot(ddxq.T, Qelmt[:,0])

  # compute H1-error
  error = L2error(Q, Gr, DGElmt, Quad2, f)**2 + L2error(dQ, Gr, DGElmt, Quad2, df)**2

  return np.sqrt(error)
  
  
  
def GoalDeriv(Q, Gr, DGElmt, Quad, f):
  """
  Derivative of J(f)=\int_{-1}^1 f(x,t)z(x,t) dx
  in direction f-Q
  f analytic solution, Q numeric approximation
  z dual analytic solution, Stepfunction: z(x,t)=f(x-1,t)
  """
  IQsq = 0.0

  for ielt in range(DGElmt.elementdofs.shape[0]):
    eltdofs = DGElmt.elementdofs[ielt]
    eltcoo  = Gr.nodecoordinates[Gr.elementnodes[ielt]]
    deltax  = eltcoo[1]-eltcoo[0]
    xq = (Quad.x+1)/2.0 * deltax + eltcoo[0]
    F  = np.zeros(xq.shape[0])
    Z  = np.zeros(xq.shape[0])
    for i in range(xq.shape[0]):
         F[i] = f(xq[i])
         Z[i] = f(xq[i]-1)
    QHelp  = np.dot(Quad.psi.T, Q[eltdofs])
    Qqsq   = np.multiply(F-QHelp[:,0],Z)

    #Integration
    IntQsq = DGElmt.J[ielt] * np.dot(Quad.w, Qqsq)
    IQsq   = IQsq + IntQsq

  return IQsq