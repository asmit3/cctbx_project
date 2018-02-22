# this script is to test how good/bad the CDF summation approximation recovers the true CDF distribution for a gaussian distribution
# with variation in sample size N
import numpy as np
import math
import matplotlib.pyplot as plt

def gauss_cdf(x,mu,sigma):
  return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

def gauss_cdf_nparray(data,mu,sigma):
  cdf = []
  for x in data:
    cdf.append(gauss_cdf(x,mu,sigma))
  return np.array(cdf)

np.random.seed(123)
mu = 0.0
sigma = 100.0
N = 25
x = np.random.normal(mu,sigma,N)
print x
x_obs = np.sort(x)
y_obs = np.array(range(1,len(x_obs)+1))/float(len(x_obs))
y_obs[:] = [z-0.5/len(x_obs) for z in y_obs]

y_calc = gauss_cdf_nparray(x_obs, mu, sigma)  
#plt.plot(x_obs,y_obs,'o')
#plt.plot(x_obs, y_calc,'g+')
#plt.show()
