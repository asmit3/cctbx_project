from __future__ import division
import scitbx.lbfgs
from scitbx.array_family import flex
from math import exp,sqrt
import math
from scitbx.matrix import sqr
import matplotlib.pyplot as plt
import numpy as np

data = """
Perform parameter fit three ways:
1) LBFGS without curvatures

Will fit these data to the following functional form(ex-gaussian distribution CDF):
y(x) = See Wikipedia

Initial values for parameters, to be refined:
24477, 131533, 175377
"""
x_obs = flex.double([])
y_obs = flex.double([])
w_obs = flex.double([])
fin = open("/net/cci-filer2/raid1/home/asmit/mcmc/real_data/LD91/merging/CDF_reflection.dat",'r')
for line in fin:
  ax = line.split()
  x_obs.append(float(ax[0]))
  y_obs.append(float(ax[1]))
  w_obs.append(1.0)

print x_obs.all()
#raw_strings = data.split("\n")
#data_index = raw_strings.index("Time Counts")
#x_obs = flex.double([float(line.split()[0]) for line in raw_strings[data_index+1:data_index+60]])
#y_obs = flex.double([float(line.split()[1]) for line in raw_strings[data_index+1:data_index+60]])
#w_obs = 1./(y_obs)

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
print initial[0], initial[1], initial[2]
#initial = flex.double([24477, 131533, 175377])

class lbfgs_exgauss ():
  def __init__(self, x_obs, y_obs, w_obs, initial):
#    super(lbfgs_exgauss,self).__init__()
#    self.set_cpp_data(x_obs,y_obs,w_obs)
    assert x_obs.size() == y_obs.size()
    self.x_obs = x_obs
    self.y_obs = y_obs
    self.w_obs = flex.double([1.0]*(w_obs.size()))
    self.n = len(initial)
    self.x = initial.deep_copy()
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self)
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
      cdf.append(self.gauss_cdf(u,0,v)-math.exp(-u+0.5*v*v+math.log(self.gauss_cdf(u,v*v,v)+1e-15)))
    return np.array(cdf)

  def exgauss_cdf(self,x, mu, sigma, tau):
    u = (x-mu)/tau
    v = sigma/tau
#    print self.gauss_cdf(u, v*v, v)
#    print self.gauss_cdf(u,0,v)-math.exp(-u+0.5*v*v+math.log(self.gauss_cdf(u,v*v,v)+1e-15))
    return self.gauss_cdf(u,0,v)-math.exp(-u+0.5*v*v+math.log(self.gauss_cdf(u,v*v,v)+1e-15))

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0,0])
    for i in range(3, x_obs.size()):
      y_calc = self.exgauss_cdf(self.x_obs[i], self.x[0], self.x[1], self.x[2]) 
      y_diff = y_obs[i] - y_calc
      result += y_diff*y_diff*self.w_obs[i]
      prefactor = -2.*w_obs[i]*y_diff
      u = (self.x_obs[i]-self.x[0])/self.x[2]
      v = self.x[1]/self.x[2]
      phi_uv2v = self.gauss_cdf(u, v*v, v) + 1e-15
#      exp_1 = np.exp(-((self.x_obs[i]-self.x[0])/(self.x[1]*np.sqrt(2)))* \
#                      ((self.x_obs[i]-self.x[0])/(self.x[1]*np.sqrt(2))))
#      exp_2 = np.exp(-((self.x_obs[i]-self.x[0]-(self.x[1]*self.x[1]/self.x[2]))/(self.x[1]*np.sqrt(2)))* \
#                      ((self.x_obs[i]-self.x[0]-(self.x[1]*self.x[1]/self.x[2]))/(self.x[1]*np.sqrt(2))))
#      if math.log(exp_1) < -50:
#        exp_1 = 0.
#      if math.log(exp_2) < -50:
#        exp_2 = 0.
      mu = self.x[0]
      sigma = self.x[1]
      tau = self.x[2]
      sqrt_2pi = np.sqrt(2*np.pi)
      z1 = (self.x_obs[i] - mu)/(sigma*np.sqrt(2))
      z2 = (self.x_obs[i]- mu-(sigma*sigma/tau))/(sigma*np.sqrt(2))
      exp_1 = np.exp(-z1*z1)
      exp_2 = np.exp(-z2*z2)
      grad[0] += prefactor*((-exp_1/(sigma*sqrt_2pi))- (1./tau - exp_2/(phi_uv2v*sigma*sqrt_2pi)))
#      from IPython import embed; embed()
      grad[1] += prefactor*((-(self.x_obs[i]-mu)*exp_1/(sigma*sigma*sqrt_2pi)) - (sigma/(tau*tau)+((sigma*(-2*sigma/tau) - \
                    (self.x_obs[i]-mu-(sigma*sigma/tau))))*(1./(sigma*sigma*phi_uv2v*sqrt_2pi))))

      grad[2] += prefactor*(-((self.x_obs[i]-mu)/(tau*tau) - (sigma*sigma)/(tau*tau*tau) + ((sigma/(tau*tau))*exp_2/(phi_uv2v*sqrt_2pi))))
#      from IPython import embed; embed(); exit()

 
#      grad[0] += (-exp_1/(self.x[1]*np.sqrt(2*np.pi))) - (1./self.x[2]+(-exp_2/(phi_uv2v*self.x[1]*np.sqrt(2*np.pi))))
#
#      grad[1] += prefactor*(-(self.x_obs[i]-self.x[0])*exp_1/(self.x[1]*self.x[1]*np.sqrt(2*np.pi))) - (self.x[1]/(self.x[2]*self.x[2])+ \
#      ((self.x[1]*np.sqrt(2)*(-2.*self.x[1]/self.x[2]) - (self.x_obs[i]-self.x[0]-(np.sqrt(2))*(self.x[1]*self.x[1]/self.x[2]))) \
#       /(2.*self.x[1]*self.x[1]))*(exp_2/(phi_uv2v*np.sqrt(np.pi))))
#
#      grad[2] += -((self.x_obs[i]-self.x[0])/(self.x[2]*self.x[2])-((self.x[1]*self.x[1])/(self.x[2]*self.x[2]*self.x[2]))+ \
#                   (self.x[1]/(self.x[2]*self.x[2]))*exp_2/(phi_uv2v*np.sqrt(2*np.pi)))
    print 'gradients', grad[0], grad[1], grad[2]
    return result,grad

  def compute_functional_and_gradients(self):
    self.a = self.x
    f,g = self.target_func_and_grad()
    self.print_step("LBFGS EXGAUSS stp",f)
    return f, g

def lbfgs_example(verbose):

#  fit = lbfgs_biexponential_fit(x_obs=x_obs,y_obs=y_obs,w_obs=w_obs,initial=initial)
  fit = lbfgs_exgauss(x_obs=x_obs,y_obs=y_obs,w_obs=w_obs,initial=initial)
  print "------------------------------------------------------------------------- "
  print "       Initial and fitted coeffcients, and inverse-curvature e.s.d.'s"
  print "------------------------------------------------------------------------- "

  for i in range(initial.size()):

    print "%2d %10.4f %10.4f %10.4f"%(
           i, initial[i], fit.a[i], sqrt(2./fit.curvatures()[i]))
  X1 = np.arange(min(x_obs), max(x_obs),1000.0)
  #from IPython import embed; embed()
  F0 = fit.exgauss_cdf_nparray(X1, initial[0], initial[1], initial[2])
  F1 = fit.exgauss_cdf_nparray(X1,fit.a[0],fit.a[1], fit.a[2])
  plt.plot(X1, F0, 'r--', linewidth=1.0)
  plt.plot(X1,F1,'g+',linewidth=0.3)
  X2 = np.sort(x_obs)
  F2 = np.array(range(len(x_obs)))/float(len(x_obs))
  plt.plot(x_obs,y_obs,'o')
  plt.show()

if (__name__ == "__main__"):
  verbose=True
  print "\n LBFGS:"
  lbfgs_example(verbose)
