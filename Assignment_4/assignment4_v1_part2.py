import openpyxl
import numpy as np
import statistics
import math

wb = openpyxl.load_workbook('Assignment_4_Data_and_Template.xlsx')
sheet = wb.get_sheet_by_name('Training Data')
last_row_number = sheet.max_row
rows_cnt = last_row_number - 1
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", rows_cnt)

###1. Initialize Data structures.
X = np.zeros((rows_cnt,15),dtype=int)
Xa = np.zeros((rows_cnt,16),dtype=int)
T1 = np.zeros((rows_cnt,1),dtype=int)
T2 = np.zeros((rows_cnt,6),dtype=int)
BinConfM = np.zeros((2,2),dtype=int)

#x = np.zeros((50,15),dtype=int)
#xa = np.zeros((50,16),dtype=int)
#t1 = np.zeros((50,1),dtype=int)
#t2 = np.zeros((50,1),dtype=int)

#2. Read XL sheet
n =0
for i in (range(2, (last_row_number+1) )):
#for i in (range(2, 11 )):
    tmp1 = [ sheet["A"+str(i)].value, sheet["B"+str(i)].value, sheet["C"+str(i)].value, sheet["D"+str(i)].value,
             sheet["E"+str(i)].value, sheet["F"+str(i)].value, sheet["G"+str(i)].value, sheet["H"+str(i)].value,
             sheet["I"+str(i)].value, sheet["J"+str(i)].value, sheet["K"+str(i)].value, sheet["L"+str(i)].value,
             sheet["M"+str(i)].value, sheet["N"+str(i)].value, sheet["O"+str(i)].value, 
             ]
    tmp2 = [1]
    tmp2.extend(tmp1)
    X[n] = tmp1
    Xa[n] = tmp2 
    T1[n] = [sheet["P"+str(i)].value]
    tmp4 = [-1, -1, -1, -1, -1, -1]
    tmp4[sheet["Q"+str(i)].value] = 1
    T2[n] = tmp4
    n = n +1
#3.
Xapi = np.linalg.pinv(Xa)
W1 = np.dot(Xapi, T1)
W2 = np.dot(Xapi, T2)


print ("Xa.shape:", Xa.shape)
print ("Xapi.shape:", Xapi.shape)
print ("W1.shape:", W1.shape)
print ("W2.shape:", W2.shape)

#**************** Performance
for i in (range(rows_cnt)):
#for i in (range(10)):
    t1_res = np.dot(Xa[i], W1)
    if   ((T1[i] < 0) and (t1_res < 0)):
        BinConfM[0][0] =  BinConfM[0][0] + 1
    elif ((T1[i] < 0) and (t1_res > 0)):
        BinConfM[0][1] =  BinConfM[0][1] + 1
    elif ((T1[i] > 0) and (t1_res < 0)):
        BinConfM[1][0] =  BinConfM[1][0] + 1
    elif ((T1[i] > 0) and (t1_res > 0)):
        BinConfM[1][1] =  BinConfM[1][1] + 1

    #n = n+1
    #print ("Xa[i]:", Xa[i], "\t", "t1_res", t1_res, "\t", "T1[i]:",  T1[i])

print ("Binary Confusion Matrix:BinConfM: \n", BinConfM)
