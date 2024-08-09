import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

inTrotterDiffDataFile="./TrotterDiffData_matrix.csv"

in_TrotterDf=pd.read_csv(inTrotterDiffDataFile,header=None)
nRow,nCol=in_TrotterDf.shape
#nRow corresponds to rows in A_inDf
#nCol corresponds to rows in dt_inDf
A_csv="./ATable.csv"
A_inDf=pd.read_csv(A_csv)
dt_csv="./dtTable.csv"
dt_inDf=pd.read_csv(dt_csv)

AVec=np.array(A_inDf.iloc[:,0])
dtVec=np.array(dt_inDf.iloc[:,0])
sorted_dtInds=np.argsort(dtVec)
sorted_dtVec=np.array([dtVec[ind] for ind in sorted_dtInds])
plt.figure()
for j in range(0,nRow):
    AVal=AVec[j]
    diffVec=np.array(in_TrotterDf.iloc[j,:])
    sorted_diffVec=[diffVec[ind] for ind in sorted_dtInds]
    plt.plot(sorted_dtVec,sorted_diffVec,label=f"A={AVal:.2e}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$dt$")
plt.ylabel("diff")
plt.legend(loc="best")
plt.title("Max diff: exact vs Trotter")

plt.savefig("diffTrotter.png")
plt.close()


