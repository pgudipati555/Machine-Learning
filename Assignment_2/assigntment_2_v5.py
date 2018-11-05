import openpyxl
import numpy as np
import numpy.linalg as npl
import statistics
import math

#************************** Function for Getting BIN Number for a given height ***************
def get_bin_num (ht_i,sp_i, ht_max, ht_min, sp_max, sp_min, bin_cnt):
    (r,c)=(0,0)
    r = round ((bin_cnt-1) * ((ht_i-ht_min)/(ht_max-ht_min)))
    c = round ((bin_cnt-1) * ((sp_i-sp_min)/(sp_max-sp_min)))
    return (r,c)

def get_pd (x, mu, cov):
    output2 = ""    
    diff = np.subtract(x,mu)
    diff_trans = diff.transpose()
    cov_det = npl.det(cov)
    cov_inv = npl.inv(cov)
    #print (output2.center(100, "^"))
    #print ("x: ", type(x), x, "\nmu:", type(mu), mu, "\ncov:", type(cov), cov)
    #print(np.shape(diff), np.shape(diff_trans))
    #print ("diff: ", type(diff), diff, "\ndiff_trans: ", type(diff_trans), diff_trans, "\ncov_det: ",type(cov_det), cov_det, "\ncov_inv: ", type(cov_inv), cov_inv)
    tmpa = np.dot(diff, cov_inv)
    tmpb = np.dot(tmpa, diff_trans)
    
    tmp1 = (1 / (2 * math.pi * math.sqrt(cov_det)) )
    tmp2 = (-0.5 * tmpb )
    tmp3 = math.exp (tmp2)
    prob_den = tmp1 * tmp3
    return(prob_den)

def bayesian (Nm, pm, Nf, pf):
    temp1 = (Nm*pm) /(Nm+Nf)
    temp2 = (Nf*pf)/(Nm+Nf)
    Pf = temp2 / (temp1+temp2)
    return (Pf)

def bayesian2 (Nm, pm, Nf, pf):
    Pf = (Nf*pf) / ((Nf*pf) + (Nm*pm) )
    return (Pf)    
    
#************* End of Functions Definitions  ***************

wb = openpyxl.load_workbook('Assignment_2_Data_and_Template.xlsx')
sheet = wb.get_sheet_by_name('Data')
last_row_number = sheet.max_row
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", (last_row_number-1))

###1. Initialize Lists and Dictionaries ****************** 
Male =[]
Female =[]
Male_And_Female = []
Bin_count = 8
H={}
Nm = 0 # Number of Male Samples
Nf = 0 # Number of female Samples

ht_max = 0 #Maximum Height overall (Male+Female)  
ht_min = 0 #Minimum Height overall (Male+Female)  
sp_max = 0 #Maximum Span overall (Male+Female)  
sp_min = 0 #Minimum Span overall (Male+Female)  

pm = 0
pf = 0

###2. Read all the rows/feature vectors from XL sheet *******************************************
output = "Read all the Rows from XL and save the hights into 3 lists: Male, Femal, Male_And_Female :"
print (output.center(100, "*"))

for i in (range(2, (last_row_number+1) )):
    fv_ht = sheet["B"+str(i)].value
    fv_sp = sheet["C"+str(i)].value
    fv_gender = sheet["A"+str(i)].value

    #print (fv_ht, fv_sp)
    Male_And_Female.append([fv_ht, fv_sp])

    if (fv_gender == "Male"):
        Male.append([fv_ht, fv_sp])
    elif (fv_gender == "Female"):
        Female.append([fv_ht, fv_sp])

M = np.array(Male)
F = np.array(Female)
MF = np.array (Male_And_Female)

#print ("Male:\n", M, "\n\n")
#print ("Female:\n", F, "\n\n")
#print ("Male and Female:\n", MF, "\n\n")

(Nm,tmp) = np.shape(M)
(Nf,tmp) = np.shape(F)

(ht_max, sp_max) = np.amax(MF, axis=0)
(ht_min, sp_min) = np.amin(MF, axis=0)

print ("Size of/Rows in M: ", Nm,"\nSize of/Rows in F:", Nf) 

if ((Nm+Nf) == (last_row_number-1)):
    print ("1. Number of Feature Vectors in the Training Set = Sum of Rowns in Male and Female Arrays- Check PASS")
else:
    print ("1. Number of Feature Vectors in the Training Set != Sum of Rowns in Male and Female Arrays- Check FAIL")

output = ""
print (output.center(30, "*"))
print ("Number of Male Samples: ",Nm)
print ("Number of Female Samples: ",Nf)
print ("Overall Max. Height: ",ht_max)
print ("Overall Min. Height: ",ht_min)
print ("Overall Max. Hand Span: ",sp_max)
print ("Overall Min. Hand Span: ",sp_min)
print ("Bin Count: ",Bin_count)
print (output.center(30, "*"))

# Create Male and Female Histograms *********************************************
output = "Create Male Histogram: "
print (output.center(100, "*"))

Hm = np.zeros((Bin_count,Bin_count), int)
Hf = np.zeros((Bin_count,Bin_count), int)

for (ht_i,sp_i) in (Male):
    (r,c) = get_bin_num(ht_i,sp_i, ht_max, ht_min, sp_max, sp_min,Bin_count)
    Hm[(int(r))][(int(c))]=Hm[(int(r))][(int(c))]+1
    #Hm[r][c]=Hm[r][c]+1

for (ht_i,sp_i) in (Female):
    (r,c) = get_bin_num(ht_i,sp_i, ht_max, ht_min, sp_max, sp_min,Bin_count)
    Hf[(int(r))][(int(c))]=Hf[(int(r))][(int(c))]+1

if ((Nm+Nf) == (np.sum(Hm)+np.sum(Hf))):
    print ("Number of Feature Vectors in the Training Set = Sum of Hm+ Sum of Hf : Check PASS")
else:
    print ("Number of Feature Vectors in the Training Set != Sum of Hm+ Sum of Hf : Check FAIL")

#****************************************************************************************
#************************************ Start of Testing **********************************
#****************************************************************************************
         
output = "Start of Testing "
output2 = ""
print (output2.center(100, "*"))
print (output.center(100, "*"))
print (output2.center(100, "*"))
tcs = [(69,17.5),(66,22),(70,21.5),(69,23.5)]

#***************** Using Histograms  ******************************
output = "Results using Histogram Method"
print (output.center(100, "~"))

for (ht_i,sp_i) in (tcs):
    (r,c) = get_bin_num(ht_i,sp_i, ht_max, ht_min, sp_max, sp_min,Bin_count)
    male_cnt = Hm[(int(r))][(int(c))]
    female_cnt = Hf[(int(r))][(int(c))]
    if ((female_cnt+male_cnt) > 0):
        p_f = (female_cnt/(female_cnt+male_cnt))
    else:
        p_f = "Undefined"
    print ("Height:", ht_i, "hand Span",sp_i, "bin number:", r,c, "Male Count:", male_cnt, "female count:", female_cnt, "Probability of being female: ", p_f, sep ="\t")

#***************** Using Bayesian Classifier  ******************************
output = "Results using Bayesian Classifier"
print (output.center(100, "~"))
MuF = np.mean(F, axis=0)
MuM = np.mean(M, axis=0)
CovF=np.cov(F,rowvar=False)
CovM=np.cov(M,rowvar=False)
print("MuF :", MuF)
print("MuM :", MuM)
print("CovF :\n", CovF)
print("CovM :\n", CovM)

#tcs = [(69,17.5)]
for (ht_i,sp_i) in (tcs):
    pd_m = get_pd (np.array([ht_i,sp_i]), MuM, CovM)
    pd_f = get_pd (np.array([ht_i,sp_i]), MuF, CovF)
    Pf1 = bayesian(Nm, pd_m, Nf, pd_f)
    Pf2 = bayesian2(Nm, pd_m, Nf, pd_f)
    print ("Height:", ht_i, "Hand Span:", sp_i, "PD_Male:", pd_m, "PD_f:", pd_f, "Pf1:", Pf1, "Pf2:", Pf2, sep ="\t")

#***************** Reconstruct Histogram Using Bayesian parameters  ******************************
output = "Reconstruct Male Histogram Using Bayesian parameters"
print (output.center(100, "~"))

Hm_re = np.zeros((Bin_count,Bin_count), )
Hf_re = np.zeros((Bin_count,Bin_count), )

ht_w = (ht_max - ht_min) / (Bin_count - 1)
sp_w = (sp_max - sp_min) / (Bin_count - 1)
#ht_w = (ht_max - ht_min) / (Bin_count)
#sp_w = (sp_max - sp_min) / (Bin_count)

for r in (range(Bin_count)):
    for c in (range(Bin_count)):
        #ht_x = ht_min + (ht_w / 2) + (ht_w * r)
        #sp_x = sp_min + (sp_w / 2) + (sp_w * c)
        ht_x = ht_min +  (ht_w * r)
        sp_x = sp_min +  (sp_w * c)
        pd_m = get_pd (np.array([ht_x,sp_x]), MuM, CovM)
        Hm_re[r][c] = round (Nm * ht_w * sp_w * pd_m)
        #print ("row:", r, "column:", c, "Coordinates: ", ht_x, sp_x, "pd_m", pd_m, "Count:", Hm_re[r][c], sep="\t" )        

output = "Reconstruct Female Histogram Using Bayesian parameters"
print (output.center(100, "~"))

for r in (range(Bin_count)):
    for c in (range(Bin_count)):
        #ht_x = ht_min + (ht_w / 2) + (ht_w * r)
        #sp_x = sp_min + (sp_w / 2) + (sp_w * c)
        ht_x = ht_min +  (ht_w * r)
        sp_x = sp_min +  (sp_w * c)
        pd_f = get_pd (np.array([ht_x,sp_x]), MuF, CovF)
        Hf_re[r][c] = round (Nf * ht_w * sp_w * pd_f)
        #print ("row:", r, "column:", c, "Coordinates: ", ht_x, sp_x,  "pd_f", pd_f, "Count:", Hf_re[r][c], sep="\t" )        

#***************** print Histograms  ******************************
output = "print Histograms "
print (output.center(100, "~"))
print ("Hm: \n",Hm)
print ("Hm_re: \n",Hm_re)
print ("Hf: \n",Hf)
print ("Hf_re: \n",Hf_re)
