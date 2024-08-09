import numpy as np
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

TrotterPath="./trotterData/"

CNPath="./CNData/"
A_csv="./ATable.csv"
A_inDf=pd.read_csv(A_csv)
dt_csv="./dtTable.csv"
dt_inDf=pd.read_csv(dt_csv)
def A_dt_rowPair_tMax(fileName):
    """

    :param fileName: data file name
    :return: A_rowNum, dt_rowNum
    """
    match_A_dtInd=re.search(r"A_rowNum(\d+)dt_rowNum(\d+)",fileName)
    match_tMatx=re.search(r"tMax([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",fileName)
    if match_A_dtInd and match_tMatx:
        A_rowNum=int(match_A_dtInd.group(1))
        dt_rowNum=int(match_A_dtInd.group(2))
        tMax=float(match_tMatx.group(1))
        return [A_rowNum,dt_rowNum,tMax]
    else:
        print("error in matching")
        exit(12)

TrotterFilesAll=[]
TrotterRowPairs_tmaxtAll=[]

for oneFile in glob.glob(TrotterPath+"/*.txt"):
    onePair=A_dt_rowPair_tMax(oneFile)
    TrotterFilesAll.append(oneFile)
    TrotterRowPairs_tmaxtAll.append(onePair)


CNFilesAll=[]
CNRowPairs_tMaxAll=[]

for oneFile in glob.glob(CNPath+"/*.txt"):
    onePair = A_dt_rowPair_tMax(oneFile)
    CNFilesAll.append(oneFile)
    CNRowPairs_tMaxAll.append(onePair)

def complex_parser(s):
    return complex(s.decode("utf-8").replace('(', '').replace(')', ''))

omegap=1

def analyticalSolution(A,t):
    """

    :param A:
    :param t:
    :return:
    """
    val=np.exp(-1j*A/omegap*np.sin(omegap*t))
    return val



def plotDiff(dataFile,A_rowNumdt_rowNumtMax):
    """

    :param dataFile:
    :param A_rowNum:
    :param dt_rowNum:
    :param tMax:
    :return:
    """
    A_rowNum,dt_rowNum,tMax=A_rowNumdt_rowNumtMax
    dataPath=os.path.dirname(dataFile)

    dtVal=dt_inDf.iloc[dt_rowNum,0]
    AVal=A_inDf.iloc[A_rowNum,0]
    M = int(np.ceil(tMax / dtVal))

    dt = tMax / M

    dataVecIn=np.genfromtxt(dataFile,dtype=complex, converters={0: complex_parser})
    dataVec=np.array([complex(val) for val in dataVecIn])
    # print(len(dataVec))
    tValsAll=np.array([dt*j for j in range(0,len(dataVec))])
    exactSolution=np.array([analyticalSolution(AVal,tj) for tj in tValsAll])
    # print(exactSolution[:3])
    diff=np.abs(dataVec-exactSolution)
    diffMax=np.max(diff)
    plt.figure()
    plt.plot(tValsAll,diff,color="blue",label="diff")
    plt.xlabel("$t$")
    plt.ylabel("diff")
    plt.title("A="+str(AVal)+", dt="+str(format(dt, ".3e")))
    plt.savefig(dataPath+"/A_rowNum"+str(A_rowNum)+"dt_rowNum"+str(dt_rowNum)+"tMax"+str(tMax)+".png")
    plt.close()
    return [A_rowNum,dt_rowNum,tMax,diffMax]

CNDiffData=[]
for n in range(0,len(CNFilesAll)):
    tPltStart=datetime.now()
    print("executing "+CNFilesAll[n])
    oneDiff=plotDiff(CNFilesAll[n],CNRowPairs_tMaxAll[n])
    tPltEnd=datetime.now()
    print("plt time: ",tPltEnd-tPltStart)
    CNDiffData.append(oneDiff)
CNPath=os.path.dirname(CNFilesAll[0])
CNDiffData=np.array(CNDiffData)
np.savetxt("./CNDiffData.txt",CNDiffData)

TrotterDiffData=[]
for n in range(0,len(TrotterFilesAll)):
    tPltStart = datetime.now()
    print("executing " + TrotterFilesAll[n])
    oneDiff =plotDiff(TrotterFilesAll[n],TrotterRowPairs_tmaxtAll[n])
    tPltEnd = datetime.now()
    print("plt time: ", tPltEnd - tPltStart)
    TrotterDiffData.append(oneDiff)

TrotterPath=os.path.dirname(TrotterFilesAll[0])
TrotterDiffData=np.array(TrotterDiffData)
np.savetxt("./TrotterDiffData.txt",TrotterDiffData)