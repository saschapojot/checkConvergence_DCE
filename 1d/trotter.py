import numpy as np
from pathlib import Path
from decimal import Decimal
import pandas as pd
import sys
from datetime import datetime
# this script generates numerical solution using Trotter decomposition

#dimension=1


psi0=1
omegap=1

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

A_rowNum=int(sys.argv[1])
dt_rowNum=int(sys.argv[2])

A_csv="./ATable.csv"
A_inDf=pd.read_csv(A_csv)

AVal=A_inDf.iloc[A_rowNum,0]

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
tEvoStart=datetime.now()
psiValsAll=[psi0]
for j in range(0,M):
    psiCurr=psiValsAll[-1]
    tj=j*dt
    psiNext=np.exp(-1j*dt*AVal*np.cos(omegap*(tj+1/2*dt)))*psiCurr
    psiValsAll.append(psiNext)
tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)
outDataFile=outPath+"/solutionTrotterA_rowNum"+str(A_rowNum)\
            +"dt_rowNum"+str(dt_rowNum)+"tMax"+str(tMax)+".txt"


np.savetxt(outDataFile,psiValsAll,delimiter=",")

