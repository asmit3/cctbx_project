import math
import numpy as np


def gauss_cdf(x,mu,sigma):
  return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

def exgauss_cdf(x, mu, sigma, tau):
  u = (x-mu)/tau
  v = sigma/tau
  return gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(gauss_cdf(u,v*v,v))

def exgauss_cdf_nparray(data,mu,sigma,tau):
  cdf = []
  for x in data:
    u = (x-mu)/tau
    v = sigma/tau
    cdf.append(gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(gauss_cdf(u,v*v,v)))
  return np.array(cdf)


from IPython import embed; embed(); exit()
