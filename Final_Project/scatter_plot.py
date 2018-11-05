import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.scattere(data[:,0],data[:,1],marker='o', facecolors='none', edgecolors='k', s=30)
x_min,x_max = min(data[:,0])-1, max(data[:, 0])+1
y_min,y_max = min(data[:,1])-1, max(data[:, 1])+1


