import numpy as np
from datetime import datetime
from scipy.special import hermite
from pathlib import Path
from scipy.misc import derivative
import numdifftools as nd
import matplotlib.pyplot as plt
import pandas as pd

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

def psi_exact(x,t):
    return f(m1,x)*np.exp(-1j*E1*t)+f(m2,x)*np.exp(-1j*E2*t)


L=5

N=2000
dx=2*L/N

tTot=10

Q=5000
dt=tTot/Q
tValsAll=[0+dt*q for q in range(0,Q+1)]
tValsAll=np.array(tValsAll)
xValsAll=[-L+2*L/N*n for n  in range(0,N)]
xValsAll=np.array(xValsAll)
xValsAll_squared=xValsAll**2

kValsAll=[2*np.pi/(2*L)*n if n<= N/2-1 else 2*np.pi/(2*L)*(n-N) for n in range(0,N)]
kValsAll=np.array(kValsAll)
kValsAll_squared=kValsAll**2
def U1(Delta_t,psiVec):
    """

    :param Delta_t: time step
    :param psiVec: psi vector in real space
    :return:
    """
    psi_hat_vec=np.fft.fft(psiVec,norm="ortho")

    k_space_evo_vec=-Delta_t*1j*1/2*kValsAll_squared

    psi_hat_vec_after_evolution=psi_hat_vec*np.exp(k_space_evo_vec)

    psi_vec_realSpace=np.fft.ifft(psi_hat_vec_after_evolution,norm="ortho")

    return psi_vec_realSpace


def U2(Delta_t,psiVec):
    """

        :param Delta_t: time step
        :param psiVec: psi vector in real space
        :return:
        """
    x_vec_evo=-Delta_t*1j*1/2*omegac**2*xValsAll_squared
    psi_vec_after_evolution=np.exp(x_vec_evo)*psiVec

    return psi_vec_after_evolution


def Phi_S(h,psiVec):
    """
    Strang splitting
    :param h:
    :param psiVec:
    :return:
    """
    psi_vec1=U1(1/2*h,psiVec)
    psi_vec2=U2(h,psi_vec1)
    psi_vec3=U1(1/2*h,psi_vec2)
    return psi_vec3






def one_step_evo(q,psiVec):
    return Phi_S(dt,psiVec)




def generate_psiExact_vec(q):
    tq=q*dt
    psiVec_exact=[psi_exact(x,tq) for x in xValsAll]
    psiVec_exact=np.array(psiVec_exact)
    psiVec_exact/=np.linalg.norm(psiVec_exact,ord=2)
    return psiVec_exact


tEvoStart=datetime.now()

diffVec=[0]
psiCurr=generate_psiExact_vec(0)
for q in range(0,100):
    print("step "+str(q))
    psiNext=one_step_evo(q,psiCurr)
    psiCurr=psiNext
    psi_analytical=generate_psiExact_vec(q+1)
    diffTmp=np.linalg.norm(psiCurr-psi_analytical)
    diffVec.append(diffTmp)

print(diffVec)

tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)


df_S=pd.DataFrame(diffVec)


outDir="./diff/"
Path(outDir).mkdir(exist_ok=True,parents=True)

outCsv_S=outDir+"/diff_S.csv"
df_S.to_csv(outCsv_S,index=False)





















