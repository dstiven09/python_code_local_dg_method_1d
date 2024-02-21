"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)
Anja Jeschke (2017)
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


def AnalyticSolution(t, x):

  g = 9.80616
  d = 10.
  #d = 3.
  a = 0.01
  l = 20.
  pi2 = 2.*np.pi
  k = pi2/l
  c = np.sqrt(g*d)
  if (A>0.):
    c = c/np.sqrt(1.+(k*d)**2/(2.*A))

  h  = d+a*np.sin(k*x)*np.cos(k*c*t)
  hu = -a*c*np.cos(k*x)*np.sin(k*c*t)

  if(A>0.):
    hw  = -0.5*d*a*k*c*np.sin(k*x)*np.sin(k*c*t)
    hpnh = -0.5*a/A*(k*c*d)**2*np.sin(k*x)*np.cos(k*c*t)
    pnh = hpnh/d
  if(A==0.):
    hw = np.zeros(len(x))
    hpnh = np.zeros(len(x))
    pnh = np.zeros(len(x))

  return h, hu, hw, pnh

#this is the variable in QQ, we want to evaluate: change it here
#ivar = 0
ivar = float(multruns_input2())
irefsolana = 1
ioutfile   = 1
ioutcons   = 0

#A = 0.
A = 1.5

def h(x):

  ht, hut, hwt, pnht = AnalyticSolution(Tmax, x)

  if(ivar==0):
    res = ht
  if(ivar==1):
    res = hut
  if(ivar==2):
    res = hwt
  if(ivar==3):
    res = pnht

  return res

def m(x):

  ht = AnalyticSolution(Tmax, x)
  return ht*ut

N    = 1       # polynomial interpolation order
xmin = 0.0    # position of left boundary
xmax = 20.0     # position of right boundary
#Tmax = 2.  # end time
Tmax = float(multruns_input())
# take only m, s.t. dx is always >=1 or <1, s.t. the log sign is the same. Otherwise you will divide through log(dx[i]/dx[i+1])=0 at the end.
#m = np.array([5, 9, 17])    # number of grid points to evaluate, sorted in ascending order
#m = np.array([6, 11, 21])
m = np.array([41, 81, 161])
#m = np.array([41, 81, 161, 321])    # number of grid points to evaluate, sorted in ascending order
mlen = len(m)
lgr, ldgel, lQQ, lh, ldx, lerrhLtwo, lerrhLinf = [], [], [], [], [], [], []


if(A==0.):
  shelp = 'swe/'
if(A==1.5):
  shelp = 'nh2/'
if(A==2.):
  shelp = 'nh1/'
expmt = 'standing/' + shelp +'standing_timestamp'
sfig  = 'diag/'+expmt+'_N='+str(N)+'_t='+str(Tmax)+'_m='
sfold = 'diag/standing/' + shelp


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
if(ivar==0):
  s_res = 'h'
if(ivar==1):
  s_res = 'hu'
if(ivar==2):
  s_res = 'hw'
if(ivar==3):
  s_res = 'hpnh'


if (ioutcons==1):
  print '###################'
  print 'Result for ', s_res, ' at Tmax = ', Tmax
  print 'L2 convergence (fitted)  : ', mhLtwo
  print 'Linf convergence (fitted): ', mhLinf

  print '\nL2 convergence'
  for i in range(mlen-1):
    print("{0:6.43f}".format(np.log(lerrhLtwo[i]/lerrhLtwo[i+1]) / np.log(ldx[i]/ldx[i+1])))

  print 'lerrhLtwo: ', lerrhLtwo

  print '\nLinf convergence'
  for i in range(mlen-1):
    print("{0:6.43f}".format(np.log(lerrhLinf[i]/lerrhLinf[i+1]) / np.log(ldx[i]/ldx[i+1])))

  print 'lerrhLinf: ', lerrhLinf


if (ioutfile==1):
  if (irefsolana==1):
    sfile = 'result_ana.out'
  else:
    sfile = 'result_num.out'
  filestr = sfold + sfile
  f = open(filestr, 'a')
  s1 = '\n###################'
  s2 = '\nResult for '+ str(s_res) + ' at Tmax = ' + str(Tmax)
  s3 = '\nL2 convergence (fitted)  : ' + str(mhLtwo)
  s4 = '\nLinf convergence (fitted): ' + str(mhLinf)

  s5 = '\n\nL2 convergence'
  f.write(s1 + s2 + s3 + s4 +s5)
  for i in range(itorun-1):
    f.write('\n' + str("{0:6.3f}".format(np.log(lerrhLtwo[i]/lerrhLtwo[i+1]) / np.log(ldx[i]/ldx[i+1]))))

  s6 = '\nlerrhLtwo: '
  f.write(s6)
  for i in range(itorun):
    f.write('\n' + str(format(lerrhLtwo[i], 'e')))

  s7 = '\n\nLinf convergence'
  f.write(s7)
  for i in range(itorun-1):
    f.write('\n' + str("{0:6.3f}".format(np.log(lerrhLinf[i]/lerrhLinf[i+1]) / np.log(ldx[i]/ldx[i+1]))))

  s8 = '\nlerrhLtwo: '
  f.write(s8)
  for i in range(itorun):
    f.write('\n' + str(format(lerrhLinf[i], 'e')))

  if (Tmax==2.0):
    f.write('\n\n\n\n')

  f.close()
