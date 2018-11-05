import openpyxl
import numpy as np
import statistics
import math

#************************** Function for Getting BIN Number for a given height ***************
def get_bin_num (i, Min, Max, bin_cnt=32):
    return (round ((bin_cnt-1) * ((i-Min)/(Max-Min))))

def get_pd (x, mu, sig):
    tmp1 = (1 / (math.sqrt(2 * math.pi) *sig))
    tmp2 = (-0.5* ( ((x - mu)/sig) *  ((x - mu)/sig) ))
    tmp3 = math.exp (tmp2)
    prob_den = tmp1 * tmp3
    return(prob_den)

def bayesian (Nm, pm, Nf, pf):
    temp1 = (Nm*pm) /(Nm+Nf)
    temp2 = (Nf*pf)/(Nm+Nf)
    Pf = temp2 / (temp1+temp2)
    return (Pf)

    
    
#************* End of Functions Definitions  ***************

wb = openpyxl.load_workbook('Assignment_1_Data_and_Template.xlsx')
sheet = wb.get_sheet_by_name('Data')
last_row_number = sheet.max_row
rows_cnt = last_row_number - 1
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", rows_cnt)

###1. Initialize Lists and Dictionaries. 
Male =[]
Female =[]
MF = []
MFC = []
Bin_count = 32
H={}
Nm = 0
Nf = 0
pm = 0
pf = 0
X = np.zeros((rows_cnt,2),dtype=int)
Xa = np.zeros((rows_cnt,2),dtype=int)
T1 = np.zeros((rows_cnt,1),dtype=int)
N = 0
Tn = 0
Tp = 0

    
j = 0
for i in (range(2, (last_row_number+1) )):
    fv_h_ft = sheet["A"+str(i)].value
    fv_h_in = sheet["B"+str(i)].value
    fv_gender = sheet["C"+str(i)].value
    
    Height = (fv_h_ft * 12 + fv_h_in)
    MF.append(Height)
    X[j][0] = Height
    Xa[j] = [1, Height]
    N = N+1

    if (fv_gender == "Male"):
        Male.append(Height)
        X[j][1] = +1
        MFC.append(+1)
        Tp = Tp+1
        T1[j] = +1
    elif (fv_gender == "Female"):
        Female.append(Height)
        X[j][1] = -1
        MFC.append(-1)
        Tn = Tn +1
        T1[j] = -1
    j = j+1

print ("N= ",N,"Tn= ",Tn,"Tp=",Tp,)

An = 0
Ap = 0
I0 = (Tp * Tn)/(N *N)
Iopt = I0
tow = MF[0]

for i in range (1, len(MF)):
    if (MFC[i-1] == -1):
        An = An +1
    elif (MFC[i-1] == +1):
        Ap = Ap +1

    tmp1 = (An*Ap)/(An+Ap);
    tmp2 = (Tn-An)*(Tp-Ap)
    tmp3 = (Tn+Tp-An-Ap)
    I = (1/N) * (tmp1 + (tmp2/tmp3))
    if (I < Iopt):
        Iopt = I
        tow = MF[i]

Delta = I0 - Iopt
print ("I0= ",I0,"Iopt= ",Iopt,"tow=",tow, "Delta: ", Delta, sep="\t")


Xapi = np.linalg.pinv(Xa)
W1 = np.dot(Xapi, T1)
