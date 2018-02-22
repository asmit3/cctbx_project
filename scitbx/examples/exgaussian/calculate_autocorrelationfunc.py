# script to calculate autocorrelation function of mu,sigma,tau values that were printed in the files params_mcmc_%seed.dat files
# that were outputted from mcmc_exgauss.py

import sys
import numpy as np
import matplotlib.pyplot as plt
seed = int(sys.argv[1])
mu = []
sigma = []
tau = []
fin = open('params_mcmc_%s.dat'%seed, 'r')
for line in fin:
  if line !='\n':
    ax = line.split()
    mu.append(float(ax[0]))
    sigma.append(float(ax[1]))
    tau.append(float(ax[2]))
# Stats for mu
y = np.array(tau)
rk = []
t = []
N = len(y)
y_bar = np.mean(y)
for k in range(0,2000,10):
  corr = 0.0
  for i in range(N-k):
    corr += (y[i]-y_bar)*(y[i+k]-y_bar)
  corr /= np.sum((y-y_bar)**2)
  rk.append(corr)
  t.append(k)

plt.plot(t, rk, 'r*', label = 'data')
plt.plot(t, [0.05]*len(t), 'k--', label ='cutoff')
#plt.plot([10]*10, [x/10.0 for x in range(10)], 'k--')
plt.xlabel('MCMC steps, k')
plt.ylabel('Correlation Coefficient C(k)')
#plt.title(r"$\frac{}{T}$")
#plt.title(r"$C(k) = \frac { \sum _{ i=1 }^{ N-k }{ \left( y_{ i }-\bar { y }  \right) *\left( y_{ i+k }-\bar { y }  \right)  }  }{ \sum _{ i=1 }^{ N }{ \left( y_{ i }-\bar { y }  \right) ^2 }  }  $", fontsize=18)
plt.show()
 
