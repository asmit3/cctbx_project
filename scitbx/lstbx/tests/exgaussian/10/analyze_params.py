# This script is for pinting out the mean(I_fit) and sigma(I_fit) from the test values generated
import numpy as np
import matplotlib.pyplot as plt

plot=True
fin = open('intensity_cdf.dat','r')
mu = []
sigma = []
tau = []
for line in fin:
  if line !='\n':
    mu.append(float(line.split()[1]))
    sigma.append(float(line.split()[2]))
    tau.append(float(line.split()[3]))

print 'Stdev of mu = %12.5f'%np.std(mu)
print 'Stdev of sigma = %12.5f'%np.std(sigma)
print 'Stdev of tau = %12.5f'%np.std(tau)

#if (plot):
#  plt.hist(intensity,normed=True, bins=20)
#  plt.show()
# Get CDF of this data
#x_obs = np.sort(intensity)
#y_obs = np.array(range(1,len(x_obs)+1))/float(len(x_obs))
#y_obs[:] = [z-0.5/len(x_obs) for z in y_obs]
#plt.figure(2)
#plt.plot(x_obs, y_obs, 'r-')
#plt.plot([71213.0]*10, [i/10.0 for i in range(10)],'o' )
#plt.show()
