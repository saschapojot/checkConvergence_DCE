import subprocess
import pandas as pd

#This script executes CN and trotter for all A and dt values


A_csv="./ATable.csv"
A_inDf=pd.read_csv(A_csv)
dt_csv="./dtTable.csv"
dt_inDf=pd.read_csv(dt_csv)

A_nRow,_=A_inDf.shape

dt_nRow,_=dt_inDf.shape

#exec trotter
for iA in range(0,A_nRow):
    for idt in range(0,dt_nRow):
        print("executing Trotter ARow="+str(iA)+", dtRow="+str(idt))
        trotterProcess=subprocess.Popen(["python3","./trotter.py", str(iA),str(idt)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            output = trotterProcess.stdout.readline()
            if output == '' and trotterProcess.poll() is not None:
                break
            if output:
                print(output.strip())
        stdout, stderr = trotterProcess.communicate()
        if stdout:
            print(stdout.strip())
        if stderr:
            print(stderr.strip())


