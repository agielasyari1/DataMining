import numpy as np
import pandas as pd
import statistics
import math
dataset= [10,12,12, 13,12,11,14,13,15,10,10, 10, 100,12, 14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10, 15,12,10,14,13,15,10]
dataset2=[10,8,10,8,8,4]
outliers=[]
def cari_outlier(data1):
    threshold=3
    rata_1=np.mean(data1)
    std_1 = np.std(data1)
    variance=statistics.variance(data1)
    std_manual=math.sqrt(variance)
    for y in data1:
        z_score = (y - rata_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
print(cari_outlier(dataset))