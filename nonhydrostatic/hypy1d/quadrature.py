"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

functions for quadrature in 1D and on the triangle

Nicole Beisiegel, April 2013
Stefan Vater, October 2013
"""

import numpy as np


def JacobiGQ(alpha, beta, N):
    """
    Compute the N'th order Gauss quadrature points, x, and weights, w,
    associated with the Jacobi polynomial of type (alpha,beta) with
    alpha, beta > -1 (!= -0.5).

    Note: adapted from Hesthaven and Warburton, 2008
    """

    from math import sqrt, gamma

    f = np.finfo(float)
    x = np.zeros(N+1)
    w = np.zeros(N+1)

    if (N==0):
        x[0] = (alpha-beta) / (alpha+beta+2.0)
        w[0] = 2.0
        return x,w

    # form symmetric matrix from recurrence.
    h1 = 2.0*np.arange(N+1) + alpha + beta
    J  = np.diag(-0.5 * (alpha**2 - beta**2)*h1 / (h1+2.0))

    for i in range(N):
        J[i,i+1] = 2.0/(h1[i]+2.0) * sqrt((i+1.0) * (i+1.0+alpha+beta) * (i+1.0+alpha) * \
                                          (i+1.0+beta) / ((h1[i]+1.0) * (h1[i]+3.0)))

    if ((alpha+beta) < f.resolution):
        J[0,0] = 0.0

    J = J + J.T

    # compute quadrature by eigenvalue solve
    [x,V] = np.linalg.eig(J)
    for j in range(N+1):
        w[j] = V[0,j]**2 * 2**(alpha+beta+1) / (alpha+beta+1.0) * \
               gamma(alpha+1) * gamma(beta+1) / gamma(alpha+beta+1)

    # obtain ascending order of Gauss points
    ind = np.argsort(x)

    return x[ind], w[ind]

#print JacobiGQ(0,0,2)


def JacobiGL(alpha, beta, N):
    """
    Compute the N'th order Gauss Lobatto quadrature points, x,
    associated with the Jacobi polynomial of type (alpha,beta) with
    alpha, beta > -1 (!= -0.5).

    Note: adapted from Hesthaven and Warburton, 2008
    """

    x = np.zeros(N+1)
    x[0] = -1.0
    x[N] =  1.0

    if (N==1):
        return x

    #[xint,w] = JacobiGQ(alpha+1, beta+1, N-2)
    xint, w = JacobiGQ(alpha + 1, beta + 1, N - 2)
    x[1:N] = xint
    return x

#print(JacobiGL(0,0,2))

