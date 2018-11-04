import openpyxl
import numpy
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
#print (type(wb))
#wb.get_sheet_names()
sheet = wb.get_sheet_by_name('Data')
last_row_number = sheet.max_row
print ("Max row Number in XL sheet=", last_row_number)
print ("Total Number of Rows in the Training set =", (last_row_number-1))

###1. Initialize Lists and Dictionaries. 
Male =[]
Female =[]
Male_And_Female = []
Bin_count = 32
H={}
Nm = 0
Nf = 0
pm = 0
pf = 0

###2.
output = "Initialise Histogram Dictionary with a size of " + str(Bin_count) + ". Index <0 thru" + str(Bin_count-1) + ">: "
print (output.center(100, "*"))
       
for i in range(0,Bin_count):
    H[i]= {'male':0, 'female':0}
#print(H)

###3.
output = "Read all the Rows from XL and save the hights into 3 lists: Male, Femal, Male_And_Female :"
print (output.center(100, "*"))
    
#for i in (range(2, 52)):
for i in (range(2, (last_row_number+1) )):
    fv_h_ft = sheet["A"+str(i)].value
    fv_h_in = sheet["B"+str(i)].value
    fv_gender = sheet["C"+str(i)].value
    
    Height = (fv_h_ft * 12 + fv_h_in)
    Male_And_Female.append(Height)

    if (fv_gender == "Male"):
        Male.append(Height)
    elif (fv_gender == "Female"):
        Female.append(Height)

#print ("Male List:",Male)
#print ("Female List:",Female)
#print ("Male and Female List:", Male_And_Female)        

Male_And_Female.sort()
Max=Male_And_Female[-1]
Min=Male_And_Female[0]
Nm = len(Male)
Nf = len(Female)
output = "Max and Minimum Height from  Male_And_Female List: "
print ("Max:", Male_And_Female[-1])
print ("Min:", Male_And_Female[0])
print ("Size of Male List:", Nm)
print ("Size of FeMale List:", Nf)
print ("Sum of both the list sizes:", (Nm+Nf))

if(((len(Male)+len(Female)) == (last_row_number -1))):
    print ("Combined length of Male and Female Lists = Number of Rows in Training Set - Check PASS")
else:
    print ("Combined length of Male and Female Lists != Number of Rows in Training Set - Check FAIL")
    
#print ("Male and Female List, after sort:", Male_And_Female)

output = "Create Male Histogram: "
print (output.center(100, "*"))

for i in (Male):
    bin_num = get_bin_num(i, Min, Max)
    H[bin_num]['male']=H[bin_num]['male']+1

output = "Create Female Histogram: "
print (output.center(100, "*"))
    
for i in (Female):
    bin_num = get_bin_num(i, Min, Max)
    H[bin_num]['female']=H[bin_num]['female']+1

output = "Mean and Standrd Deviation for Male and Female Histograms: "
print (output.center(100, "*"))
  
Mu_M = statistics.mean(Male)
Mu_F = statistics.mean(Female)

sig_M = statistics.stdev(Male)
sig_F = statistics.stdev(Female)

print ("Mean Male:",Mu_M)
print ("Mean Female:",Mu_F)
print ("stdev Male:",sig_M)
print ("stdev Female:",sig_F)

output2 = ""
print (output2.center(100, "~"))
print ("Counts in each Bin:", H);
print (output2.center(100, "~"))

output = "Check whether the Sum of all Buckets is Equal to Number of Rows in Training Set : "
print ("Max:", Male_And_Female[-1])
print ("Min:", Male_And_Female[0])
count_h = 0
for i in (H):
    count_h = count_h + H[i]['male']+H[i]['female']


if ( count_h == (last_row_number -1)):
         print ("The Sum of all Buckets is Equal to Number of Rows in Training Set - Check PASS")
else:
         print ("The Sum of all Buckets is Equal to Number of Rows in Training Set - Check FAIL")

#****************************************************************************************
#************************************ Start of Testing **********************************
#****************************************************************************************
         
output = "Start of Testing "
output2 = ""
print (output2.center(100, "*"))
print (output.center(100, "*"))
print (output2.center(100, "*"))
tcs = [55,60,65,70,75,80]

#***************** Using Histograms  ******************************
output = "Results using Histogram Method"
print (output.center(100, "~"))

for ht_in in (tcs):
    bin_num = get_bin_num(ht_in, Min, Max)
    male_cnt = H[bin_num]['male']
    female_cnt = H[bin_num]['female']
    p_f = (female_cnt/(female_cnt+male_cnt))
    print ("Height:", ht_in, "bin number:", bin_num, "Male Count:", male_cnt, "female count:", female_cnt, "Probability of being female: ", p_f, sep ="\t")

#***************** Using Gaussian Model  ******************************
output = "Results using Gaussian Method"
print (output.center(100, "~"))

for ht_in in (tcs):
    pd_m = get_pd (ht_in, Mu_M, sig_M)
    pd_f = get_pd (ht_in, Mu_F, sig_F)
    Pf = bayesian(Nm, pd_m, Nf, pd_f)
    print ("Height:", ht_in, "PD_Male:", pd_m, "PD_f:", pd_f, "Pf:", Pf, sep ="\t")
