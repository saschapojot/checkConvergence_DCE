import numpy as np
from datetime import datetime
from scipy.special import hermite

from scipy.misc import derivative
import numdifftools as nd
import matplotlib.pyplot as plt



omegac=10
def H(n,x):
    """

    :param n: order of Hermite polynomial
    :param x:
    :return: value of polynomial at x
    """
    return hermite(n)(x)


def f(n,x):
    return np.exp(-1 / 2 * omegac * x ** 2) * H(n, np.sqrt(omegac) * x)

def E(n):
    return 1/2*omegac*(2*n+1)

m1=2

m2=5

E1=E(m1)
E2=E(m2)

def psi(x,t):
    return f(m1,x)*np.exp(-1j*E1*t)+f(m2,x)*np.exp(-1j*E2*t)



def psi_real(x, t):
    return np.real(psi(x,t))

def psi_imag(x, t):
    return np.imag(psi(x,t))

#########################################################################
# check pde
# x0=1
# t0=2
#
# dx_2_real = nd.Derivative(lambda x: psi_real(x, t0), n=2)
# d_x2_imag = nd.Derivative(lambda x: psi_imag(x, t0), n=2)
#
#
# dt_real=nd.Derivative(lambda t:psi_real(x0,t),n=1)
# dt_imag=nd.Derivative(lambda t: psi_imag(x0,t),n=1)
#
# dt_val=dt_real(t0)+1j*dt_imag(t0)
#
# lhs=1j*dt_val
#
# dx2_val=dx_2_real(x0)+1j*d_x2_imag(x0)
#
# rhs=-1/2*dx2_val+1/2*omegac**2*x0**2*psi(x0,t0)
#
# diff=lhs-rhs
#
# print(diff)

#########################################################################


###################################
# plt
L=5

N=2000
dx=2*L/N

xValsAll=[-L+dx*n for n in range(0,N)]
t=1

absPsiValsAll=[np.abs(psi(x,t)) for x in xValsAll]

plt.figure()
plt.plot(xValsAll,absPsiValsAll,color="black")
plt.savefig("tmp.png")
plt.close()





###################################