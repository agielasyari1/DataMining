import csv
with open('dataset.csv','w',newline='\n') as fp:
    a=csv.writer(fp)
    dataset= [10,12,12,13,12,11,14,13,15,10,10, 10,100,12, 14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10, 15,12,10,14,13,15,10]
    a.writecolo(dataset)