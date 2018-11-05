import openpyxl
import os
import struct
import matplotlib as plt
from array import array as pyarray
from pylab import *
from numpy import *
import numpy.linalg as la
import statistics
import math


wb = openpyxl.load_workbook('Assignment_3_ Submission_Results_Prasanth.xlsx')
sheet = wb.get_sheet_by_name('Results')

xrow = 2
for xcol in (range (2, 11)): 
    sheet.cell(row=xrow, column=xcol).value = (2000+xcol)
    wb.save('Assignment_3_ Submission_Results_Prasanth.xlsx')
