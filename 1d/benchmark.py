import numpy as np

import glob
import re


TrotterPath="./trotterData/"

CNPath="./CNData/"


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

def complex_converter(s):
    return complex(s.replace('j', '1j'))