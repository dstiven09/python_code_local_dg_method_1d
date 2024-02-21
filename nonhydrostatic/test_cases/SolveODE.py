"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

Test for time stepping method applied to a simple ODE
"""

import matplotlib.pyplot as plt
import numpy as np

from hypy1d.timestepping    import RungeKutta

def RHS(u,t):
  return -u

def main():
  """
  main program
  """

  RK1 = RungeKutta(RHS, 1)
  RK2 = RungeKutta(RHS, 2)
  RK3 = RungeKutta(RHS, 3)
  u1 = [1.0]
  u2 = [1.0]
  u3 = [1.0]
  t  = [0.0]
  tmax = 10.0
  dt = 0.5

  for tt in np.arange(0.0, tmax, dt):
    unew = RK1.step(u1[-1], tt, dt)
    u1.append(unew)
    unew = RK2.step(u2[-1], tt, dt)
    u2.append(unew)
    unew = RK3.step(u3[-1], tt, dt)
    u3.append(unew)
    t.append(tt+dt)

  texact = np.linspace(0, tmax, 200)
  uexact = np.exp(-texact)

  fig = plt.figure(1)
  fig.clf()
  ax1 = fig.add_subplot(111)
  ax1.plot(t, u1, label='RK1')
  ax1.plot(t, u2, label='RK2')
  ax1.plot(t, u3, label='RK3')
  ax1.plot(texact, uexact, label='exact')
  ax1.legend()
  plt.show(block=False)

if __name__ == "__main__":
  main()
