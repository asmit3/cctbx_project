from __future__ import division
import scitbx.lbfgs
from scitbx.array_family import flex
from math import exp,sqrt
import math
from scitbx.matrix import sqr
import matplotlib.pyplot as plt
import numpy as np
import sys
from minimize_exgauss import lbfgs_exgauss
from test3_logarithm import sampler
data = """
Perform parameter fit three ways:
1) LBFGS without curvatures

Will fit these data to the following functional form(ex-gaussian distribution CDF):
cdf(x) = gauss_cdf(u,0,v)-exp(-u+0.5*v*v+log(gauss_cdf(u,v*v,v)
u = (x-mu)/tau
v = sigma/tau
"""
# Ex-Gaussian Data
x_obs = []
y_obs = []
w_obs = []
fin = open(sys.argv[1],'r')
for line in fin:
  miller_index = '0_0_0'
  if '#' in line:
    ax = line.split()
    miller_index = '%s_%s_%s'%(ax[-3],ax[-2],ax[-1])
  else:
    ax = line.split()
    if float(ax[0]) > -1.e15 and float(ax[0]) < 1.e15:
      x_obs.append(float(ax[0]))
      w_obs.append(1.0)
fin.close()

# Adjust CDF to make it (n-0.5)/N instead of n/N to create a buffer
x_obs = np.sort(x_obs)
y_obs = np.array(range(1,len(x_obs)+1))/float(len(x_obs))
y_obs[:] = [z-0.5/len(x_obs) for z in y_obs]

def skewness(data):
  y_bar = np.mean(data)
  s = np.mean(data)
  N=len(data)
  g = np.sum((data-y_bar)*(data-y_bar)*(data-y_bar))
  g = g/(N*s*s*s)
  return g

def initial_guess(data):
  data = np.array(data)
  mu = np.mean(data) - skewness(data)
  tau = np.std(data)*0.8
  sigma = np.sqrt(np.var(data)-tau*tau)
  return [mu,sigma,tau]

initial = flex.double(initial_guess(x_obs))

# Make it flex doubles
x_obs = flex.double(x_obs)
y_obs = flex.double(y_obs)
w_obs = flex.double(w_obs)

fit = lbfgs_exgauss(x_obs=x_obs,y_obs=y_obs,w_obs=w_obs,initial=initial)
print "------------------------------------------------------------------------- "
print "       Initial and fitted coeffcients, and inverse-curvature e.s.d.'s"
print "------------------------------------------------------------------------- "

for i in range(initial.size()):
  print "%2d %10.4f %10.4f %10.4f"%(
  i, initial[i], fit.a[i], fit.a[i])
  X1 = x_obs.as_numpy_array()

# =========== NOW DO THE MCMC bit here below ==========
print '======================= MCMC stuff beginning ============================'
nsteps = 500
x_obs = x_obs.as_numpy_array()
maxI = np.max(x_obs)
minI = np.min(x_obs)
proposal_width =  0.01*np.abs(maxI-minI)
print 'initial guesses and proposal width = ',fit.a[0], fit.a[1], fit.a[2], proposal_width
params = sampler(x_obs, samples=nsteps, mu_init= fit.a[0],sigma_init = fit.a[1],tau_init = fit.a[2],
                   proposal_width = proposal_width,plot=False)

mu,sigma, tau = params[-1]
print 'final parameter values ',mu,sigma, tau



