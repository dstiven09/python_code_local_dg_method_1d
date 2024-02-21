"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
"""

import matplotlib.pyplot as plt
import numpy as np
from math import degrees, atan

from hypy1d.grid            import Grid1D, generate_uniformgrid
from hypy1d.dg_element      import DGReferenceElement1D, DGElement1D, DGProlong, DGQuadrature1D, L2error, Linferror
from hypy1d.interpolation   import Vandermonde1D

def multruns_input():

  value  = np.loadtxt('multruns_in')

  return value

def multruns_input2():

  value  = np.loadtxt('multruns_in2')

  return value


# interpolation from first data points onto second data points
def func_interpolate(x1,x2,y1):
  y2 = np.zeros(len(x2))
  for i in range(len(x2)):
    m = 0.
    for j in range(len(x1)-1):
      if ((x1[j]<=x2[i]) and (x2[i]<=x1[j+1])):
        m = (y1[j+1]-y1[j])/(x1[j+1]-x1[j])
        break
    y2[i] = m*(x2[i]-x1[j])+y1[j]
  return y2

def AnalyticSolution(x):

  a = 1.

  ## Diri0
  #h = a*np.ones(len(x))                    # h: h
  #q = np.cos(np.pi*(x+1.))*(np.pi)/a       # q: hu (analyt. sol. for a=const, not used)
  #f = -np.sin(np.pi*(x+1.))*(np.pi**2)     # rhs in 2. equation: hw
  #u = np.sin(np.pi*(x+1.))/a**2            # u: pnh ( analyt. sol. for a=const, not used)

  ## refl:
  h = a*np.ones(len(x))                    # h: h
  q = -np.sin(np.pi*(x+1.))*(np.pi)/a      # q: hu (analyt. sol. for a=const, not used)
  f = -np.cos(np.pi*(x+1.))*(np.pi**2)     # rhs in 2. equation: hw
  u = np.cos(np.pi*(x+1.))/(a**2) -1.      # u: pnh ( analyt. sol. for a=const, not used)


  #h   = np.cos(np.pi*(x+1.))*(np.pi)/a        # h: h
  #h_x = -np.sin(np.pi*(x+1.))*(np.pi**2)/a
  #q   = h**2                                  # q: hu (analyt. sol. for a=const, not used)
  #f   = 2.*q*h_x                              # rhs in 2. equation: hw
  #u   = np.sin(np.pi*(x+1.))/a                # u: pnh ( analyt. sol. for a=const, not used)

  #h    = (2.+np.cos(np.pi*(x+1.)))/a            # h: h
  #u_x  = np.cos(np.pi*(x+1.))/a*np.pi
  #u_xx = -np.sin(np.pi*(x+1.))/a*(np.pi)**2
  #h_x  = -np.sin(np.pi*(x+1.))/a*(np.pi)
  #q_x  = h*u_xx + h_x*u_x
  #q    = h*u_x                             # q: hu (analyt. sol. for a=const, not used)
  #f    = h*q_x                             # rhs in 2. equation: hw
  #u    = (0.+np.sin(np.pi*(x+1.)))/a            # u: pnh ( analyt. sol. for a=const, not used)



  return q, u

ivar = float(multruns_input2())
irefsolana = 1
ioutfile   = 1
ioutcons   = 0

def h(x):

  q, u = AnalyticSolution(x)

  if(ivar==1):
    res = q
  if(ivar==3):
    res = u

  return res

def m(x):

  ht, ut, hwt, hpnht = AnalyticSolution(x)
  #here and down in the code you have to set 'h' to the variable you want to evaluate
  return ht*ut

N    = 1      # polynomial interpolation order
xmin = -1.0    # position of left boundary
xmax = 1.0  # position of right boundary
# take only m, s.t. dx is always >=1 or <1, s.t. the log sign is the same. Otherwise you will divide through log(dx[i]/dx[i+1])=0 at the end.
m = np.array([3, 5, 9, 17, 33, 65, 129, 257, 513])    # number of grid points to evaluate, sorted in ascending order
#m = np.array([41, 81, 161, 321, 641])    # number of grid points to evaluate, sorted in ascending order
mlen = len(m)
lgr, ldgel, lQQ, lh, ldx, lerrhLtwo, lerrhLinf = [], [], [], [], [], [], []

expmt = 'poisson/poisson'
sfig  = 'diag/convergence/'+expmt+'_N='+str(N)+'_m='
sfold = 'diag/convergence/poisson/'

# checking the periodic analytic solution
#j = 2
#xlen = xmax - xmin
#x = np.arange(0.,xlen, xlen/(m[j]-1.))
#fig=plt.figure(1)
#plt.plot(x, h(x), color='b', linestyle='-', linewidth=3, label='h')
#figname = "poisson_t=" + str(Tmax) + ".eps"
#plt.savefig(figname, format='eps', dpi=800)
##quit()


relt  = DGReferenceElement1D(N)
dgqu  = DGQuadrature1D(relt, 5)


if (irefsolana==1):
  itorun = mlen
else:
  itorun = mlen-1


for i in range(mlen):
  ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, m[i], periodic=True)
  gr = Grid1D(ndcoo, eltnd, ndels)
  dgel = DGElement1D(gr, relt)
  #load data
  QQ = np.loadtxt(sfig+str(float(m[i]))+'.out')
  lgr.append(gr)
  ldgel.append(dgel)
  lQQ.append(QQ[:,ivar])

for i in range(mlen-1):
  href = DGProlong(lQQ[i], relt, ldgel[i], ldgel[-1])
  lh.append(href)
lh.append(lQQ[mlen-1])

for i in range(itorun):
  dx = (xmax-xmin)/(float(m[i])-1.0)
  ldx.append(dx)


for i in range(itorun):
  if (irefsolana==1):
    Ltwo = L2error(lh[i], lgr[-1], ldgel[-1], dgqu, h, irefsolana)
    Linf = Linferror(lh[i], lgr[-1], ldgel[-1], relt, 11, h, irefsolana)
  else:
    Ltwo = L2error(lh[i], lgr[-1], ldgel[-1], dgqu, lh[-1], irefsolana)
    Linf = Linferror(lh[i], lgr[-1], ldgel[-1], relt, 11, lh[-1], irefsolana)
  lerrhLtwo.append(Ltwo)
  lerrhLinf.append(Linf)


arrdx = np.log(ldx[0:itorun])
A   = np.vstack([arrdx, np.ones(len(arrdx))]).T

mhLtwo, chLtwo = np.linalg.lstsq(A, np.log(lerrhLtwo[:]))[0]
mhLinf, chLinf = np.linalg.lstsq(A, np.log(lerrhLinf[:]))[0]




# output of results
if(ivar==1):
  s_res = 'q'
if(ivar==3):
  s_res = 'u'

if (ioutcons==1):
  print '###################'
  print 'Result for ', s_res
  print 'L2 convergence (fitted)  : ', mhLtwo
  print 'Linf convergence (fitted): ', mhLinf

  print '\nL2 convergence'
  for i in range(mlen-1):
    print("{0:6.4f}".format(np.log(lerrhLtwo[i]/lerrhLtwo[i+1]) / np.log(ldx[i]/ldx[i+1])))

  print 'lerrhLtwo: ', lerrhLtwo

  print '\nLinf convergence'
  for i in range(mlen-1):
    print("{0:6.4f}".format(np.log(lerrhLinf[i]/lerrhLinf[i+1]) / np.log(ldx[i]/ldx[i+1])))

  print 'lerrhLinf: ', lerrhLinf


if (ioutfile==1):
  if (irefsolana==1):
    sfile = 'result_ana.out'
  else:
    sfile = 'result_num.out'
  filestr = sfold + sfile
  f = open(filestr, 'a')
  s1 = '\n###################'
  s2 = '\nResult for '+ str(s_res)
  s3 = '\nL2 convergence (fitted)  : ' + str(mhLtwo)
  s4 = '\nLinf convergence (fitted): ' + str(mhLinf)

  s5 = '\n\nL2 convergence'
  f.write(s1 + s2 + s3 + s4 +s5)
  for i in range(itorun-1):
    f.write('\n' + str("{0:6.3f}".format(np.log(lerrhLtwo[i]/lerrhLtwo[i+1]) / np.log(ldx[i]/ldx[i+1]))))

  #s6 = '\nlerrhLtwo: ' + str(lerrhLtwo)
  s6 = '\nlerrhLtwo: '
  f.write(s6)
  for i in range(itorun):
    f.write('\n' + str(format(lerrhLtwo[i], 'e')))

  s7 = '\n\nLinf convergence'
  f.write(s7)
  for i in range(itorun-1):
    f.write('\n' + str("{0:6.3f}".format(np.log(lerrhLinf[i]/lerrhLinf[i+1]) / np.log(ldx[i]/ldx[i+1]))))

  #s8 = '\nlerrhLinf: ' + str(lerrhLinf)
  s8 = '\nlerrhLtwo: '
  f.write(s8)
  for i in range(itorun):
    f.write('\n' + str(format(lerrhLinf[i], 'e')))

  f.write('\n\n\n\n')

  f.close()
