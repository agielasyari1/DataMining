import numpy as np
import pandas as pd
import statistics
import math
setdata= [10,12,12, 13,12,11,14,13,15,10,10, 10, 20,12, 14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10, 15,12,10,14,13,15,10]
outliers=[]
def metode_std(data):
    rata=np.mean(data)
    std=np.std(data)
    lebiDari=rata+(2.75*std)
    kurangDari=rata-(2.75*std)
    for y in data:
        if y >= lebiDari:
            outliers.append(y)
        elif y<=kurangDari:
            outliers.append(y)
    return outliers
print(metode_std(setdata))

