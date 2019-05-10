import math
import numpy as np
import pandas as pd
plot1 = [1,2,3,20,2]
plot2 = [2,1,1,30,2]
z=[]
a={}
print()
phi=0,5
r=len(plot)-1
phin=0,5*len(plot1)
outlier=[]
df=pd.DataFrame({'x': plot1, 'y' : plot2})
def distance (data1,data2):
   for i in range (len(plot1)):
      for j in range (len(plot1)):
         jarak = math.sqrt((data1[j] - data1[i]) ** 2 +(data2[j]- data2[i]) **2)
           a[i+1, j+1] = jarak
           z.append(jarak)
   for k in z:
      if k >= phin:
         outlier.appand(k)

distance(plot1,plot2)
print(df)
print("phi , n=",phi)
print("r = ",r)
print("hasil =",aa)
print("outlier =",outlier)