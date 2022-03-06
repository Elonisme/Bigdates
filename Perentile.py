import numpy as np
X = np.array([-35,10,20,30,40,50,60,100])
k=25
Xk = np.percentile(X, k,method= 'linear')
Nx = X.shape[0]
indices = 1 + (Nx - 1)*k/100.0
print(indices,Xk)