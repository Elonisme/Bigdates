import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
plt.grid(True, linestyle='-.',color = "black", linewidth="0.2")
Samples = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
                    [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]])
                    
mean_x = np.mean(Samples[0,:])
mean_y = np.mean(Samples[1,:])
mean_vector = np.array([[mean_x],[mean_y]])
Samples_zero_mean = Samples - mean_vector
plt.scatter(Samples_zero_mean[0], Samples_zero_mean[1],color = 'b')
plt.show()
# 零均值化

Cov_Samples_zero_mean = Samples_zero_mean.dot(Samples_zero_mean.T)/9;
print(Cov_Samples_zero_mean)
# 样本协方差

#计算特征值和特征向量
eig_val, eig_vec = np.linalg.eig(Cov_Samples_zero_mean)
print(eig_val)
print(eig_vec)

#可视化特征向量
plt.scatter(0, 0, marker = '.', color = 'r')
plt.scatter(Samples_zero_mean[0], Samples_zero_mean[1])
plt.arrow(0, 0, eig_vec.T[0,0], eig_vec.T[0,1], head_width = 0.02, head_length = 0.1, fc = 'r', ec = 'r')
plt.arrow(0, 0, eig_vec.T[1,0], eig_vec.T[1,1], head_width = 0.02, head_length = 0.1, fc = 'r', ec = 'r')
plt.show()
