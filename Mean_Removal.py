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