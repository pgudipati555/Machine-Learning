import numpy as np
import numpy.linalg as LA

X = np.array([[72.,101.,94.],[50.,96.,70.],[14.,79.,10.],[8.,70.,1.]], np.float64);
print(X)

Mu=np.mean(X,axis=0);print(Mu)
Z=X-Mu;print(Z)

C=np.cov(Z,rowvar=False);print(C)

[Lambda,V]=LA.eigh(C);print(Lambda,'\n\n',V)



