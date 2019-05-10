import numpy as np
import pandas as pd
import statistics
import math
#dataset= [1,2,2,3,2,1,4,13,5,10,10,10,500,2,4,3,2,10,10,11,12,5,2,3,2,1,4,3,5,10,5,2,10,4,3,1,10]
local = pd.read_csv("D:\OneDrive\Documents\Kuliah S1\Semester 4\Data Mining\untitled\dataset.csv")
ubahlist=list(local)
dataset=[]
for i in ubahlist:
   float(i)
   dataset.append(int(float(i)))
outliers=[]
def metodeiqr(data):
    q1,q3=np.percentile(sorted(data),[25,75])
    b_bawah=q1-(1.5*q1)
    b_atas=q3+(1.5*q3)
    for y in data:
        if y <=b_bawah:
            outliers.append(y)
        elif y >=b_atas:
            outliers.append(y)
    return outliers
print(metodeiqr(dataset))