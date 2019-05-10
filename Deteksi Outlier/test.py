import numpy as np
import pandas as pd
import statistics as st
dataset= [10,12,12, 13,12,11,14,13,15,10,10, 10, 100,12, 14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10, 15,12,10,14,13,15,10]
outlier_iqr=[]
outlier_std=[]
outlier_jarak=[]
def menu(data):
    print('''Progam sederhana pencari data outlier''')
    print('''[1] Metode distance based\n[2] Metode IQR\n[3] Metode Standar deviasi\n[4] Semua metode''')
    x = input("masukan pilihan anda:")
    if x=="2":
        metode_IQR(data)
    elif x == "3":
        metode_std(data)
    elif x=="4":
        print("Hasil dari metode IQR:"),metode_IQR(data)
        print("Hasil dari metode STD:"),metode_std(data)
def metode_IQR(data):
    q1,q3=np.percentile(sorted(data),[25,75])
    batas_bawah = q1 - (1.5 * q1)
    batas_atas = q3 + (1.5 * q3)
    for y in data:
        if y <= batas_bawah:
            outlier_iqr.append(y)
        elif y >= batas_atas:
            outlier_iqr.append(y)
    for i in outlier_iqr:
        print(i)
def metode_std(data):
    rata=np.mean(data)
    std=np.std(data)
    lebiDari=rata+(2.75*std)
    kurangDari=rata-(2.75*std)
    for y in data:
        if y >= lebiDari:
            outlier_std.append(y)
        elif y<=kurangDari:
            outlier_std.append(y)
    for i in outlier_std:
        print(i)
menu(dataset)