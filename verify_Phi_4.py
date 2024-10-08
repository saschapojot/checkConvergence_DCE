import numpy as np
from datetime import datetime
from scipy.special import hermite
from pathlib import Path
# from scipy.misc import derivative
# import numdifftools as nd
import matplotlib.pyplot as plt
import pandas as pd



gamma1_3=1/(2-2**(1/3))
gamma2_3=-2**(1/3)/(2-2**(1/3))
gamma3_3=1/(2-2**(1/3))



##########################################
# copied from verify_Phi_S.py
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

Q=10000
dt=tTot/Q
tValsAll=[0+dt*q for q in range(0,Q+1)]
tValsAll=np.array(tValsAll)
xValsAll=[-L+2*L/N*n for n  in range(0,N)]
xValsAll=np.array(xValsAll)
xValsAll_squared=xValsAll**2

kValsAll=[2*np.pi/(2*L)*n if n<= N/2-1 else 2*np.pi/(2*L)*(n-N) for n in range(0,N)]
kValsAll=np.array(kValsAll)
kValsAll_squared=kValsAll**2
def U1_S(Delta_t,psiVec):
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


def U2_S(Delta_t,psiVec):
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
    psi_vec1=U1_S(1/2*h,psiVec)
    psi_vec2=U2_S(h,psi_vec1)
    psi_vec3=U1_S(1/2*h,psi_vec2)
    return psi_vec3

def generate_psiExact_vec(q):
    tq=q*dt
    psiVec_exact=[psi_exact(x,tq) for x in xValsAll]
    psiVec_exact=np.array(psiVec_exact)
    psiVec_exact/=np.linalg.norm(psiVec_exact,ord=2)
    return psiVec_exact
##########################################


def Phi_4(h,psiVec):
    psi_vec1=Phi_S(gamma3_3*h,psiVec)

    psi_vec2=Phi_S(gamma2_3*h,psi_vec1)

    psi_vec3=Phi_S(gamma1_3*h,psi_vec2)

    return psi_vec3



tEvoStart=datetime.now()

diffVec=[0]
psiCurr=generate_psiExact_vec(0)
for q in range(0,Q):
    print("step "+str(q))
    psiNext =Phi_4(dt,psiCurr)
    psiCurr = psiNext
    psi_analytical = generate_psiExact_vec(q + 1)
    diffTmp = np.linalg.norm(psiCurr - psi_analytical)
    diffVec.append(diffTmp)

tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)

df_Phi_4=pd.DataFrame(diffVec)
outDir="./diff/"
Path(outDir).mkdir(exist_ok=True,parents=True)
outCsv_Phi_4=outDir+"/diff_Phi_4.csv"
df_Phi_4.to_csv(outCsv_Phi_4,index=False)