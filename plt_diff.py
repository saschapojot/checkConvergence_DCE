import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


inDir="./diff/omegac10/"

inFile_Phi_S=inDir+"/diff_S.csv"

inFile_Phi_4=inDir+"/diff_Phi_4.csv"

inFile_Phi_6=inDir+"/diff_Phi_6.csv"

inFile_Phi_8=inDir+"/diff_Phi_8.csv"


diff_Phi_S=pd.read_csv(inFile_Phi_S,header=None)

diff_Phi_4=pd.read_csv(inFile_Phi_4,header=None)

diff_Phi_6=pd.read_csv(inFile_Phi_6,header=None)

diff_Phi_8=pd.read_csv(inFile_Phi_8,header=None)


tTot=10

Q=10000
dt=tTot/Q

tValsAll=[dt*q for q in range(0,len(diff_Phi_4))]

plt.figure()
plt.plot(tValsAll,diff_Phi_S,label="Phi_S")
plt.plot(tValsAll,diff_Phi_4,label="Phi_4")
plt.plot(tValsAll,diff_Phi_6,label="Phi_6")
plt.plot(tValsAll,diff_Phi_8,label="Phi_8")
plt.yscale("log")
plt.legend(loc="best")
plt.savefig(inDir+"/diff.png")

plt.close()