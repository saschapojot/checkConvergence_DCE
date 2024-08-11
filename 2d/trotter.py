import numpy as np
from pathlib import Path
from decimal import Decimal
import pandas as pd
import sys
from datetime import datetime
import pickle
# this script generates numerical solution using Trotter decomposition

#dimension=2

psi0=np.array([1,1])/np.sqrt(2)
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

A1_rowNum=int(sys.argv[1])
dt_rowNum=int(sys.argv[2])


A1_csv="./A1Table.csv"
A1_inDf=pd.read_csv(A1_csv)
A1Val=A1_inDf.iloc[A1_rowNum,0]

dt_csv="./dtTable.csv"
dt_inDf=pd.read_csv(dt_csv)
dtVal=dt_inDf.iloc[dt_rowNum,0]

outPath="./trotterData/"
Path(outPath).mkdir(parents=True, exist_ok=True)


def format_using_decimal(value):
    # Convert the float to a Decimal
    decimal_value = Decimal(value)
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


tMax=100

#t steps
M=int(np.ceil(tMax/dtVal))

dt=tMax/M
print("dt="+str(dt))
tEvoStart=datetime.now()
psiValsAll=np.zeros((M+1,2),dtype=complex)
psiValsAll[0,:]=psi0
A0Val=1e3
omegap=1
def coef0(t):
    return np.exp(-1j*dt*A0Val*np.cos(omegap*(t+1/2*dt)))
def coef1(t):
    return np.exp(-1j*dt*A1Val*np.cos(omegap*(t+1/2*dt)))
tEvoStart=datetime.now()

for j in range(0,M):
    tj=dt*j
    psiCurr=psiValsAll[j,:]
    psiNext=[coef0(tj)*psiCurr[0],coef1(tj)*psiCurr[1]]
    psiValsAll[j+1,:]=psiNext

tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)
outDataFile=outPath+"/solutionTrotterA1_rowNum"+str(A1_rowNum)\
            +"dt_rowNum"+str(dt_rowNum)+"tMax"+str(tMax)+".pkl"

with open(outDataFile,"wb") as fptr:
    pickle.dump(psiValsAll,fptr)



