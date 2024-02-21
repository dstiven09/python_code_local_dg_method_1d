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

#A = 0.
A = 1.5

def AnalyticSolution(t, x):

  xlen = xmax-xmin
  g = 9.80616
  d = 10.
  a = 2.
  K = np.sqrt(3.*a/(4.*d*d*(d+a)))
  c = np.sqrt(g*(d+a))
  x0 = xlen/4.

  nperiod = int(t*c/xlen)+2
  ssh = np.zeros(len(x))
  hw  = np.zeros(len(x))
  hpnh= np.zeros(len(x))

  if (A>0.):
    for i in range(nperiod):
      xhelp = x - x0 -c*t + float(i)*xlen
      ssh_help = a/((np.cosh(K*xhelp))**2)
      T_help = np.tanh(K*xhelp)
      h_help = d + ssh_help
      terms_help = (2.*(T_help**2)*(d/h_help)-ssh_help/a)/(d+ssh_help)
      hpnh_help = ((d*c*K)**2)*ssh_help/A*terms_help

      ssh = ssh + ssh_help
      hw = hw + d*c*K*ssh_help*T_help
      hpnh = hpnh + hpnh_help

    h    = d + ssh
    hu   = c*ssh
    pnh  = hpnh

  if (A==0.):
    c = np.sqrt(g*d)
    numf = 1000
    xf = np.zeros(numf)
    for i in range(numf):
      xf[i] = xlen/float(numf-1)*float(i)
    xhelp2 = np.zeros((nperiod,len(xf)))
    for i in range(nperiod):
      xhelp2[i] = xf-c*0.-x0+float(i)*xlen

    fact = float(int(t/30.))
    for i in range(nperiod):
      sshf = a/((np.cosh(K*xhelp2[0]))**2)
      etaf = xhelp2[i]-fact*xlen -x0+(3.*np.sqrt(g*(d+sshf))-2.*np.sqrt(g*d))*t
      ssh = ssh + func_interpolate(etaf, x, sshf)

    h = d + ssh
    hu  = h*2.*(np.sqrt(g*(d+ssh)) - c)
    hw  = np.zeros(len(x))
    hpnh = np.zeros(len(x))

  return h, hu, hw, hpnh

#this is the variable in QQ, we want to evaluate: change it here
#ivar = 0
ivar = float(multruns_input2())
irefsolana = 1
ioutfile   = 1
ioutcons   = 0
iCFLtest   = 1 # =1 if CFL=const; =0 if dx=const

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

N    = 1      # polynomial interpolation order
xmin = 0.0    # position of left boundary
xmax = 800.0  # position of right boundary
#Tmax = 20.   # end time
Tmax = float(multruns_input())
m = np.array([101, 201, 401, 801])    # number of grid points to evaluate, sorted in ascending order
#m = np.array([401, 801, 1601, 3201, 6401, 12801])    # number of grid points to evaluate, sorted in ascending order
mlen = len(m)
lgr, ldgel, lQQ, lh, ldx, lerrhLtwo, lerrhLinf = [], [], [], [], [], [], []

if(A==0.):
  shelp = 'swe/'
if(A==1.5):
  shelp = 'nh2/'
if(A==2.):
  shelp = 'nh1/'
srun = 'solitary'
#srun = 'solitary_local'
expmt = srun + '/' + shelp + srun + '_timestamp'
sfig  = 'diag/'+expmt+'_N='+str(N)+'_t='+str(Tmax)+'_m='
sfold = 'diag/' + srun + '/' + shelp

# checking the periodic analytic solution
#j = 2
#xlen = xmax - xmin
#x = np.arange(0.,xlen, xlen/(m[j]-1.))
#fig=plt.figure(1)
#plt.plot(x, h(x), color='b', linestyle='-', linewidth=3, label='h')
##axes = plt.gca()
##axes.set_ylim([10.-0.2,12.2])
##figname = "simple_period_t=" + str(Tmax) + ".eps"
#figname = "solitary_period_t=" + str(Tmax) + ".eps"
#plt.savefig(figname, format='eps', dpi=800)
##quit()


relt  = DGReferenceElement1D(N)
dgqu  = DGQuadrature1D(relt, 5)


if (irefsolana==1):
  itorun = mlen
else:
  itorun = mlen-1


for i in range(mlen):
  if (iCFLtest==1):
    n = m[i]
  else:
    n = m[0]
  ndcoo, eltnd, ndels = generate_uniformgrid(xmin, xmax, n, periodic=True)
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
  s_res = 'pnh'


if (ioutcons==1):
  print '###################'
  print 'Result for ', s_res, ' at Tmax = ', Tmax
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

  if (Tmax==20.0):
    f.write('\n\n\n\n')

  f.close()
