import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

QStr="1e6"
omegac=10000
inDir=f"./diff/Q{QStr}/omegac{omegac}/"

# inFile_Phi_S=inDir+"/diff_S.csv"
inFile_Phi_6=inDir+"/diff_Phi_6.csv"

inFile_Phi_8=inDir+"/diff_Phi_8.csv"

inFile_Phi_10=inDir+"/diff_Phi_10.csv"

# inFile_Phi_12=inDir+"/diff_Phi_12.csv"
#
# inFile_Phi_14=inDir+"/diff_Phi_14.csv"
#
# inFile_Phi_16=inDir+"/diff_Phi_16.csv"


# diff_Phi_S=pd.read_csv(inFile_Phi_S,header=None)
diff_Phi_6=pd.read_csv(inFile_Phi_6,header=None)

diff_Phi_8=pd.read_csv(inFile_Phi_8,header=None)



diff_Phi_10=pd.read_csv(inFile_Phi_10,header=None)

# diff_Phi_12=pd.read_csv(inFile_Phi_12,header=None)
#
# diff_Phi_14=pd.read_csv(inFile_Phi_14,header=None)
#
# diff_Phi_16=pd.read_csv(inFile_Phi_16,header=None)


tTot=10

Q=float(QStr)
dt=tTot/Q

tValsAll=[dt*q for q in range(0,len(diff_Phi_10))]

plt.figure()
# plt.plot(tValsAll,diff_Phi_S,label="Phi_S")
plt.plot(tValsAll,diff_Phi_6,label="Phi_6")
plt.plot(tValsAll,diff_Phi_8,label="Phi_8")
# plt.plot(tValsAll,diff_Phi_10,label="Phi_10")
# plt.plot(tValsAll,diff_Phi_12,label="Phi_12")
# plt.plot(tValsAll,diff_Phi_14,label="Phi_14")
# plt.plot(tValsAll,diff_Phi_16,label="Phi_16")
plt.title(r"$\omega_{c}=$"+str(omegac))
plt.yscale("log")
plt.legend(loc="best")
plt.savefig(inDir+"/diff.png")

plt.close()