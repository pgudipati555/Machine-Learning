import openpyxl
import numpy as np
import statistics
import math

wb = openpyxl.load_workbook('Data_User_Modeling_Dataset_keslerized_data.xlsx')
sheet = wb.get_sheet_by_name('Training_Data')
last_row_number = sheet.max_row
rows_cnt = last_row_number - 1
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", rows_cnt)

###1. Initialize Data structures.
print ("1. Initialize Data structures.".center(100, "~"))
X = np.zeros((rows_cnt,5),dtype=float)
Xa = np.ones((rows_cnt,6),dtype=float)
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
for i in (range(2, (last_row_number+1) )):
    X[n][0] = sheet["A"+str(i)].value
    X[n][1] = sheet["B"+str(i)].value
    X[n][2] = sheet["C"+str(i)].value
    X[n][3] = sheet["D"+str(i)].value
    X[n][4] = sheet["E"+str(i)].value  
#
    T2[n][0] = sheet["G"+str(i)].value
    T2[n][1] = sheet["H"+str(i)].value
    T2[n][2] = sheet["I"+str(i)].value
    T2[n][3] = sheet["J"+str(i)].value
#
    T2_truth.append(sheet["F"+str(i)].value)
#
    n = n +1
#2.2
for i in (range(len(X))):
	Xa[i][1:] =X[i]

#3. Veify the counts of each feature:

#4. Calculate Inverse matrix and find W2.
print ("Calculate W2".center(100, "~"))
Xapi = np.linalg.pinv(Xa)
W2 = np.dot(Xapi, T2)
print ("X.shape",       X.shape, sep="\t")
print ("Xa.shape",      Xa.shape, sep="\t")
print ("W2.shape",      W2.shape, sep="\t")

#5 **************** Calculate 4 class Confusion Matrix
print ("*** Calculate 4 Class Confusion Matrix *****")
for i in (range(rows_cnt)):

    t2_temp = list(np.dot(Xa[i], W2))
    t2_class = t2_temp.index( max(t2_temp))

    if (T2_truth[i]=="very_low"):
        if (t2_class == 0):
            SixConfM[0][0] =  SixConfM[0][0] + 1
        elif (t2_class == 1):
            SixConfM[0][1] =  SixConfM[0][1] + 1
        elif (t2_class == 2):
            SixConfM[0][2] =  SixConfM[0][2] + 1
        elif (t2_class == 3):
            SixConfM[0][3] =  SixConfM[0][3] + 1
    elif (T2_truth[i]=="High"):
        if (t2_class == 0):
            SixConfM[1][0] =  SixConfM[1][0] + 1
        elif (t2_class == 1):
            SixConfM[1][1] =  SixConfM[1][1] + 1
        elif (t2_class == 2):
            SixConfM[1][2] =  SixConfM[1][2] + 1
        elif (t2_class == 3):
            SixConfM[1][3] =  SixConfM[1][3] + 1
    elif (T2_truth[i]=="Low"):
        if (t2_class == 0):
            SixConfM[2][0] =  SixConfM[2][0] + 1
        elif (t2_class == 1):
            SixConfM[2][1] =  SixConfM[2][1] + 1
        elif (t2_class == 2):
            SixConfM[2][2] =  SixConfM[2][2] + 1
        elif (t2_class == 3):
            SixConfM[2][3] =  SixConfM[2][3] + 1
    elif (T2_truth[i]=="Middle"):
        if (t2_class == 0):
            SixConfM[3][0] =  SixConfM[3][0] + 1
        elif (t2_class == 1):
            SixConfM[3][1] =  SixConfM[3][1] + 1
        elif (t2_class == 2):
            SixConfM[3][2] =  SixConfM[3][2] + 1
        elif (t2_class == 3):
            SixConfM[3][3] =  SixConfM[3][3] + 1
            
print ("Very_low", "High", "Low", "Middle", "<-----Classified-Class", sep="\t");
print (SixConfM, "\n\n")

