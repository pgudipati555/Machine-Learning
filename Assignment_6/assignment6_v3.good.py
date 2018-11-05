import openpyxl
import numpy as np
import statistics
import math

wb = openpyxl.load_workbook('Car_Data.xlsx')
sheet = wb.get_sheet_by_name('Sheet1')
last_row_number = sheet.max_row
rows_cnt = last_row_number - 2
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", rows_cnt)

###1. Initialize Data structures.
print ("1. Initialize Data structures.".center(100, "~"))
X = np.zeros((rows_cnt,21),dtype=int)
Xa = np.ones((rows_cnt,22),dtype=int)
T1 = np.zeros((rows_cnt,1),dtype=int)
T2 = np.zeros((rows_cnt,4),dtype=int)
#T2_truth = np.array((rows_cnt,1),dtype=str)
T2_truth = [];
BinConfM = np.zeros((2,2),dtype=int)
SixConfM = np.zeros((4,4),dtype=int)

for i in (range(len(T2))):
	T2[i] = [-1, -1, -1, -1]

for i in (range(len(T2))):
	T1[i] = [-1]
	
#2. Read XL sheet
print ("2. Read XL sheet into X, Xa and T2".center(100, "~"))
n =0
for i in (range(3, (last_row_number+1) )):
    
    if (sheet["A"+str(i)].value == "low"):
        X[n][0] = 1
    elif (sheet["A"+str(i)].value == "med"):
        X[n][1] = 1
    elif (sheet["A"+str(i)].value == "high"):
        X[n][2] = 1
    elif (sheet["A"+str(i)].value == "vhigh"):
        X[n][3] = 1    
#
    if (sheet["B"+str(i)].value == "low"):
        X[n][4] = 1
    elif (sheet["B"+str(i)].value == "med"):
        X[n][5] = 1
    elif (sheet["B"+str(i)].value == "high"):
        X[n][6] = 1
    elif (sheet["B"+str(i)].value == "vhigh"):
        X[n][7] = 1
#
    if (sheet["C"+str(i)].value == 2):
        X[n][8] = 1
    elif (sheet["C"+str(i)].value == 3):
        X[n][9] = 1
    elif (sheet["C"+str(i)].value == 4):
        X[n][10] = 1
    elif (sheet["C"+str(i)].value == "5more"):
        X[n][11] = 1
#
    if (sheet["D"+str(i)].value == 2):
        X[n][12] = 1
    elif (sheet["D"+str(i)].value == 4):
        X[n][13] = 1
    elif (sheet["D"+str(i)].value == "more"):
        X[n][14] = 1
#
    if (sheet["E"+str(i)].value == "small"):
        X[n][15] = 1
    elif (sheet["E"+str(i)].value == "med"):
        X[n][16] = 1
    elif (sheet["E"+str(i)].value == "big"):
        X[n][17] = 1 
#
    if (sheet["F"+str(i)].value == "low"):
        X[n][18] = 1
    elif (sheet["F"+str(i)].value == "med"):
        X[n][19] = 1
    elif (sheet["F"+str(i)].value == "high"):
        X[n][20] = 1 
###
    if (sheet["G"+str(i)].value == "unacc"):
        T2[n][0] = 1
    elif (sheet["G"+str(i)].value == "acc"):
        T2[n][1] = 1
    elif (sheet["G"+str(i)].value == "good"):
        T2[n][2] = 1
    elif (sheet["G"+str(i)].value == "vgood"):
        T2[n][3] = 1    
#
    if ((sheet["G"+str(i)].value == "acc") or (sheet["G"+str(i)].value == "good") or (sheet["G"+str(i)].value == "vgood")):
        T1[n][0] = 1
#
    T2_truth.append(sheet["G"+str(i)].value)
#
    n = n +1
#2.2
for i in (range(len(X))):
	Xa[i][1:] =X[i]

#3. Veify the counts of each feature:
print ("3. Veify the counts of each feature from DataStructures".center(100, "~"))
sums = list(X.sum(axis=0))
sumf1 = sums[0]+sums[1]+sums[2]+sums[3]
sumf2 = sums[4]+sums[5]+sums[6]+sums[7]
sumf3 = sums[8]+sums[9]+sums[10]+sums[11]
sumf4 = sums[12]+sums[13]+sums[14]
sumf5 = sums[15]+sums[16]+sums[17]
sumf6 = sums[18]+sums[19]+sums[20]

print ("X.shape", X.shape, sep="\t")
print ("Xa.shape", Xa.shape, sep="\t")
print ("T2.shape", T2.shape, sep="\t")
print ("sumf1", sumf1, "sumf2", sumf2,"sumf3", sumf3,"sumf4", sumf4,"sumf5", sumf5,"sumf6", sumf6,sep="\t")

if ((rows_cnt==sumf1) and (rows_cnt==sumf2) and (rows_cnt==sumf3) and (rows_cnt==sumf4) and (rows_cnt==sumf5) and (rows_cnt==sumf6) ):
    print ("All Counts are matching: *** PASS ***".center(80, "-"))
else:
    print ("All Counts are NOT matching: *** FAIL ***".center(80, "-"))

#4
Xapi = np.linalg.pinv(Xa)
W2 = np.dot(Xapi, T2)
print ("W2.shape", W2.shape, sep="\t")


#5 **************** Calculate 4 class Confusion Matrix
print ("*** Calculate 4 Class Confusion Matrix *****")
for i in (range(rows_cnt)):

    t2_temp = list(np.dot(Xa[i], W2))
    t2_class = t2_temp.index( max(t2_temp))

    if (T2_truth[i]=="unacc"):
        if (t2_class == 0):
            SixConfM[0][0] =  SixConfM[0][0] + 1
        elif (t2_class == 1):
            SixConfM[0][1] =  SixConfM[0][1] + 1
        elif (t2_class == 2):
            SixConfM[0][2] =  SixConfM[0][2] + 1
        elif (t2_class == 3):
            SixConfM[0][3] =  SixConfM[0][3] + 1
    elif (T2_truth[i]=="acc"):
        if (t2_class == 0):
            SixConfM[1][0] =  SixConfM[1][0] + 1
        elif (t2_class == 1):
            SixConfM[1][1] =  SixConfM[1][1] + 1
        elif (t2_class == 2):
            SixConfM[1][2] =  SixConfM[1][2] + 1
        elif (t2_class == 3):
            SixConfM[1][3] =  SixConfM[1][3] + 1
    elif (T2_truth[i]=="good"):
        if (t2_class == 0):
            SixConfM[2][0] =  SixConfM[2][0] + 1
        elif (t2_class == 1):
            SixConfM[2][1] =  SixConfM[2][1] + 1
        elif (t2_class == 2):
            SixConfM[2][2] =  SixConfM[2][2] + 1
        elif (t2_class == 3):
            SixConfM[2][3] =  SixConfM[2][3] + 1
    elif (T2_truth[i]=="vgood"):
        if (t2_class == 0):
            SixConfM[3][0] =  SixConfM[3][0] + 1
        elif (t2_class == 1):
            SixConfM[3][1] =  SixConfM[3][1] + 1
        elif (t2_class == 2):
            SixConfM[3][2] =  SixConfM[3][2] + 1
        elif (t2_class == 3):
            SixConfM[3][3] =  SixConfM[3][3] + 1
            
print ("Four Class Confusion Matrix:SixConfM: \n", SixConfM)
#**************** Calculate Binary Confusion Matrix
print ("*** Calculate Binary Confusion Matrix *****")
W1 = np.dot(Xapi, T1)

for i in (range(rows_cnt)):
    t1_res = np.dot(Xa[i], W1)
    if   ((T1[i] < 0) and (t1_res < 0)):
        BinConfM[0][0] =  BinConfM[0][0] + 1
    elif ((T1[i] < 0) and (t1_res > 0)):
        BinConfM[0][1] =  BinConfM[0][1] + 1
    elif ((T1[i] > 0) and (t1_res < 0)):
        BinConfM[1][0] =  BinConfM[1][0] + 1
    elif ((T1[i] > 0) and (t1_res > 0)):
        BinConfM[1][1] =  BinConfM[1][1] + 1

print ("Binary Confusion Matrix:BinConfM: \n", BinConfM)
print ("W2: \n", W2)

