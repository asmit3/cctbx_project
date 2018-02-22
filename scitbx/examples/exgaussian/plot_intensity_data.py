import sys
import matplotlib.pyplot as plt
import numpy as np
fin = open(sys.argv[1],'r')
x_obs = []
y_obs = []
F0 = []
F1 = []
for line in fin:
  ax = line.split()
#  x_obs.append(np.log(float(ax[0])))
  x_obs.append(float(ax[0]))
  y_obs.append(float(ax[1]))
  F0.append(float(ax[2]))
  F1.append(float(ax[3]))

plt.plot(x_obs,y_obs,'.')
plt.plot(x_obs, F0, 'r*', linewidth=2.0)
plt.plot(x_obs,F1,'g+',linewidth=2)
plt.show()
