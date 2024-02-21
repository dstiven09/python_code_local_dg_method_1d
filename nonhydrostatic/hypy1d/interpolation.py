"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

functions for interpolation in 1D

Nicole Beisiegel, April 2013
Stefan Vater, October 2013
"""

import numpy as np


def JacobiP(x, alpha, beta, N):
    """
    Evaluate the Jacobi polynomial of type (alpha, beta), alpha, beta > -1,
    (alpha+beta) != -1, at points x for order N and returns P(x). The
    polynomials are normalized to be orthonormal.

    Note: adapted from Hesthaven and Warburton, 2008
    """

    from math import sqrt, gamma

    PL = np.zeros((N+1, np.size(x)))

    # initial values P_0(x) and P_1(x)
    gamma0  = 2**(alpha+beta+1) / (alpha+beta+1.0) * \
              gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1)

    PL[0,:] = 1.0 / sqrt(gamma0)

    if(N == 0):
        return PL[0]

    gamma1  = (alpha+1.0) * (beta+1.0) / (alpha+beta+3.0) * gamma0
    PL[1,:] = ((alpha+beta+2.0)*x/2.0 + (alpha-beta)/2.0) / sqrt(gamma1)

    if(N == 1):
        return PL[1]

    # repeat value in recurrence
    aold = 2.0 / (2.0+alpha+beta) * sqrt((alpha+1.0)*(beta+1.0) / (alpha+beta+3.0))

    # forward recurrence using the symmetry of the recurrence
    for i in range(1,N):
        h1   = 2.0*i + alpha + beta
        anew = 2.0/(h1+2.0) * sqrt((i+1.0) * (i+1.0+alpha+beta) * (i+1.0+alpha) * \
                                   (i+1.0+beta) / ((h1+1.0)*(h1 + 3.0)))
        bnew = - (alpha**2 - beta**2) / (h1*(h1+2.0))
        PL[i+1] = 1.0/anew * (-aold*PL[i-1] + (x-bnew)*PL[i])
        aold = anew

    return PL[N]


def Vandermonde1D(N,r):
    """
    Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i)

    Note: adapted from Hesthaven and Warburton, 2008
    """

    V1D = np.zeros((np.size(r), N+1))

    for j in range(N+1):
        V1D[:,j]= JacobiP(r,0,0,j)

    return V1D


def GradJacobiP(r, alpha, beta, N):
    """
    Evaluate the derivative of the Jacobi polynomial of type (alpha,beta) with
    alpha, beta > -1, at points r for order N and returns dP[1:length(r))]

    Note: adapted from Hesthaven and Warburton, 2008
    """

    from math import sqrt

    if(N == 0):
        return np.zeros(np.size(r))
    else:
        return sqrt(N * (N+alpha+beta+1.0)) * JacobiP(r, alpha+1, beta+1, N-1)


def GradVandermonde1D(N, r):
    """
    Initialize the gradient of the modal basis (phi(i)) up to order N at
    points r

    Note: adapted from Hesthaven and Warburton, 2008
    """

    DVr = np.zeros((np.size(r), N+1))

    # Initialize matrix
    for i in range(N+1):
        DVr[:,i] = GradJacobiP(r, 0, 0, i)

    return DVr


def Dmatrix1D(N, r, V):
    """
    Initialize the differentiation matrix on the interval,
    evaluated at (r) at order N (using the Vandermonde matrix V)

    Note: adapted from Hesthaven and Warburton, 2008
    """

    # asserts:  Dr.T is skew-antisymmetric as long as r_i = r_{N-i},
    # each row-sum of Dr.T is exactly zero,
    # Dr.T is nilpotent

    Vr = GradVandermonde1D(N, r)
    Dr = np.linalg.solve(V.T, Vr.T)

    return Dr


#JacobiP(np.array((-1,0,1)), 0, 0, 2)
#print(Vandermonde1D(2,np.array((-1., 0., 1.))))