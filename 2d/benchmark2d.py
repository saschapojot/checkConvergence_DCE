import numpy as np
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
A1_csv="./A1Table.csv"
A1_inDf=pd.read_csv(A1_csv)
dt_csv="./dtTable.csv"
dt_inDf=pd.read_csv(dt_csv)



def A1_dt_rowPair_tMax(fileName):
    """

    :param fileName: data file name
    :return: A1_rowNum, dt_rowNum
    """
    match_A1_dtInd=re.search(r"A1_rowNum(\d+)dt_rowNum(\d+)",fileName)
    match_tMatx=re.search(r"tMax([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",fileName)
    if match_A1_dtInd and match_tMatx:
        A1_rowNum=int(match_A1_dtInd.group(1))
        dt_rowNum=int(match_A1_dtInd.group(2))
        tMax=float(match_tMatx.group(1))
        return [A1_rowNum,dt_rowNum,tMax]
    else:
        print("error in matching")
        exit(12)

TrotterPath="./trotterData/"
TrotterFilesAll=[]
TrotterRowPairs_tmaxtAll=[]

for oneFile in glob.glob(TrotterPath+"/*.pkl"):
    onePair=A1_dt_rowPair_tMax(oneFile)
    TrotterFilesAll.append(oneFile)
    TrotterRowPairs_tmaxtAll.append(onePair)

omegap=1
A0Val=1e3
psi0=np.array([1,1])/np.sqrt(2)

def analyticalSolution(A0,A1,t):
    """
    :param A0:
    :param A1:
    :param t:
    :return:
    """

    val0=psi0[0]*np.exp(-1j*A0/omegap*np.sin(omegap*t))
    val1=psi0[1]*np.exp(-1j*A1/omegap*np.sin(omegap*t))
    return np.array([val0,val1])




def plotDiff(dataFile,A1_rowNumdt_rowNumtMax):
    """

    :param dataFile:
    :param A1_rowNum:
    :param dt_rowNum:
    :param tMax:
    :return:
    """
    A1_rowNum, dt_rowNum, tMax = A1_rowNumdt_rowNumtMax
    dataPath = os.path.dirname(dataFile)
    dtVal = dt_inDf.iloc[dt_rowNum, 0]

    A1Val = A1_inDf.iloc[A1_rowNum, 0]
    M = int(np.ceil(tMax / dtVal))
    dt = tMax / M

    with open(dataFile,"rb") as fptr:
        psiArray=pickle.load(fptr)
    nRow,_=psiArray.shape
    tValsAll = np.array([dt*j for j in range(0,nRow)])
    exactSolution = np.array([analyticalSolution(A0Val,A1Val,tj) for tj in tValsAll] )
    exactSolution=np.array(exactSolution)
    diffArray=exactSolution-psiArray
    diff=np.linalg.norm(diffArray,axis=1,ord=2)
    diffMax = np.max(diff)
    plt.figure()
    plt.plot(tValsAll, diff, color="blue", label="diff")
    plt.xlabel("$t$")
    plt.ylabel("diff")
    plt.title("A0=" + str(A0Val)+", A1="+str(A1Val) + ", dt=" + str(format(dt, ".3e")))
    plt.savefig(dataPath+"/A1_rowNum"+str(A1_rowNum)+"dt_rowNum"+str(dt_rowNum)+"tMax"+str(tMax)+".png")
    plt.close()
    return [A1_rowNum,dt_rowNum,tMax,diffMax]

TrotterDiffData=[]
for n in range(0,len(TrotterFilesAll)):
    tPltStart = datetime.now()
    print("executing " + TrotterFilesAll[n])
    oneDiff= plotDiff(TrotterFilesAll[n],TrotterRowPairs_tmaxtAll[n])
    tPltEnd = datetime.now()
    print("plt time: ", tPltEnd - tPltStart)
    TrotterDiffData.append(oneDiff)

TrotterPath=os.path.dirname(TrotterFilesAll[0])
TrotterDiffData=np.array(TrotterDiffData)
np.savetxt("./TrotterDiffData.txt",TrotterDiffData)