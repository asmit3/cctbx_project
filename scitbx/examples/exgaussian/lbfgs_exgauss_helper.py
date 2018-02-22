from __future__ import division
import scitbx.lbfgs
from scitbx.array_family import flex
from math import exp,sqrt
import math
from scitbx.matrix import sqr
import matplotlib.pyplot as plt
import numpy as np
import sys
from test3_logarithm import *
data = """
Perform parameter fit and then does an mcmc optimization:
1) LBFGS without curvatures for initial parameter fitting
2) MCMC for phase space expoloration of the parameters

Following Idea of Sharma/Neutze(2017)
Will fit these data to the following functional form(ex-gaussian distribution CDF):
cdf(x) = gauss_cdf(u,0,v)-exp(-u+0.5*v*v+log(gauss_cdf(u,v*v,v)
u = (x-mu)/tau
v = sigma/tau
"""
#x_obs = flex.double([])
#y_obs = flex.double([])
#w_obs = flex.double([])
# Make up some fake data from an ex-gaussian distribution

class lbfgs_exgauss ():
  def __init__(self, x_obs, y_obs, w_obs, initial):
    assert x_obs.size() == y_obs.size()
    self.x_obs = x_obs
    self.y_obs = y_obs
    self.w_obs = w_obs #flex.double([1.0]*(w_obs.size()))
    self.n = len(initial)
    self.x = initial.deep_copy()
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self,
                     termination_params = scitbx.lbfgs.termination_parameters
                     (traditional_convergence_test_eps=1.e-10, min_iterations=0), core_params =
                     scitbx.lbfgs.core_parameters(gtol=0.00011), log=sys.stdout)
    self.a = self.x

  def skewness(self,data):
    y_bar = np.mean(data)
    s = np.mean(data)
    N=len(data)
    g = np.sum((data-y_bar)*(data-y_bar)*(data-y_bar))
    g = g/(N*s*s*s)
    return g

  def initial_guess(self,data):
    data = np.array(data)
    mu = np.mean(data) - self.skewness(data)
    tau = np.std(data)
    sigma = np.sqrt(np.var(data)-tau*tau)

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target),
    print "["," ".join(["%10.4f"%a for a in pfh.x]),"]"

  def curvatures(self):
    return flex.double([1, 1, 1])

# Good blog on different minimizers
# http://aria42.com/blog/2014/12/understanding-lbfgs
  def gauss_cdf(self,x,mu,sigma):
    return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

  def exgauss_cdf_nparray(self,data,mu,sigma,tau):
    cdf = []
    for x in data:
      u = (x-mu)/tau
      v = sigma/tau
      if self.gauss_cdf(u,v*v,v) == 0.0:
        cdf.append(self.gauss_cdf(u,0,v))
      else:
        cdf.append(self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v)))
    return np.array(cdf)

  def exgauss_cdf(self,x, mu, sigma, tau):
    u = (x-mu)/tau
    v = sigma/tau
#    print u,v,self.gauss_cdf(u,0,v), self.gauss_cdf(u,v*v,v)
    if self.gauss_cdf(u,v*v,v) == 0.0:
      return self.gauss_cdf(u,0,v)
    else:
      return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v))

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0,0])
    for i in range(0, self.x_obs.size()):
      y_calc = self.exgauss_cdf(self.x_obs[i], self.x[0], self.x[1], self.x[2]) 
      if np.abs(y_calc) > 1.0:
#        continue
        print 'exgaussian overflow',i, self.x_obs[i], self.x[0], self.x[1], self.x[2]
#        from IPython import embed; embed(); exit()
      y_diff = self.y_obs[i] - y_calc
      result += y_diff*y_diff*self.w_obs[i]
      prefactor = -2.*self.w_obs[i]*y_diff
      mu = self.x[0]
      sigma = self.x[1]
      tau = self.x[2]
      u = (self.x_obs[i]-mu)/tau
      v = sigma/tau
      phi_uv2v = self.gauss_cdf(u, v*v, v) 
      z1 = (self.x_obs[i] - mu)/(sigma*np.sqrt(2))
      z2 = (self.x_obs[i]- mu-(sigma*sigma/tau))/(sigma*np.sqrt(2))
      exp_1 = np.exp(-z1*z1)/np.sqrt(2*np.pi*sigma*sigma)
      exp_2 = np.exp(-z2*z2)/np.sqrt(np.pi)
      exp_0 = np.exp(-u+0.5*v*v)
      if np.isinf(exp_0):
        exp_0 = 0.0
      grad[0] += prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2))))))
      grad[1] += prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((phi_uv2v*sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma))))
      grad[2] += prefactor*((0.0) - exp_0*((phi_uv2v*u/tau) - (phi_uv2v*v*v/tau) + (exp_2*(v/(tau*np.sqrt(2))))))

#    print 'gradients', grad[0], grad[1], grad[2]
    return result,grad

  def compute_functional_and_gradients(self):
    self.a = self.x
    f,g = self.target_func_and_grad()
    self.print_step("LBFGS EXGAUSS stp",f)
    return f, g

  def curvatures(self):
    f,g = self.target_func_and_grad()
    curvs = flex.double([1., 1., 1.])
    return curvs
    

def mcmc_lbfgs_example(verbose):
  if (exgauss):
    fit = lbfgs_exgauss(x_obs=x_obs,y_obs=y_obs,w_obs=w_obs,initial=initial)
  else:
    fit = lbfgs_gauss(x_obs=x_obs,y_obs=y_obs,w_obs=w_obs,initial=initial)
  print "------------------------------------------------------------------------- "
  print "       Initial and fitted coeffcients, and inverse-curvature e.s.d.'s"
  print "------------------------------------------------------------------------- "

  for i in range(initial.size()):
    print "%2d %10.4f %10.4f %10.4f"%(
           i, initial[i], fit.a[i], fit.a[i])
  X1 = x_obs.as_numpy_array()
  plt.figure(1)
  plt.plot(x_obs,y_obs,'.')#, facecolors='None',edgecolors='b')
#  fout = open('exgauss_simulated.dat','w')
#  for i in range(len(X1)):
#    fout.write("%12.4f" %X1[i])
#    fout.write("%12.4f\n" %y_obs[i])
  F0 = fit.exgauss_cdf_nparray(X1, initial[0], initial[1], initial[2])
  F1 = fit.exgauss_cdf_nparray(X1,fit.a[0],fit.a[1], fit.a[2])
  F2 = fit.exgauss_cdf_nparray(X1,-4000.0,4000.0, 25000.0)
  plt.plot(X1,F1,'g+',linewidth=2)
#  plt.plot(X1,F2,'k-',linewidth=2)
  print 'Sum Squared Difference = ',sum(map(lambda x:x*x,F1-y_obs))

  from construct_random_datapt import ExGauss
  EXG= ExGauss(10000, np.min(x_obs), np.max(x_obs), fit.a[0], fit.a[1], fit.a[2])
  cdf_cutoff = 0.95
  I_fit = EXG.interpolate_x_value(cdf_cutoff)
  print I_fit

if (__name__ == "__main__"):
  verbose=True
  print "\n LBFGS:"
  mcmc_lbfgs_example(verbose)
