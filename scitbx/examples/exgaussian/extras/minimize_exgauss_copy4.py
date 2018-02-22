from __future__ import division
import scitbx.lbfgs
from scitbx.array_family import flex
from math import exp,sqrt
import math
from scitbx.matrix import sqr
import matplotlib.pyplot as plt
import numpy as np
import sys

data = """
Perform parameter fit three ways:
1) LBFGS without curvatures

Will fit these data to the following functional form(ex-gaussian distribution CDF):
cdf(x) = gauss_cdf(u,0,v)-exp(-u+0.5*v*v+log(gauss_cdf(u,v*v,v)
u = (x-mu)/tau
v = sigma/tau
"""
#x_obs = flex.double([])
#y_obs = flex.double([])
#w_obs = flex.double([])
# Make up some fake data from an ex-gaussian distribution
def clamp(x):
  if x < 0:
    return x
  else:
    return x

def gauss_cdf(x,mu,sigma):
  return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

def gauss_cdf_nparray(data,mu,sigma):
  cdf = []
  for x in data:
    cdf.append(clamp(gauss_cdf(x,mu,sigma)))
  return np.array(cdf)

def exgauss_cdf_nparray(data,mu,sigma,tau):
  cdf = []
  for x in data:
    u = (x-mu)/tau
    v = sigma/tau
    cdf.append(clamp(gauss_cdf(u,0,v)-math.exp(-u+0.5*v*v+np.log(gauss_cdf(u,v*v,v)+1e-15))))
  return np.array(cdf)

exgauss = True
is_fake_exgauss = False
# Ex-Gaussian Data
if (exgauss):
  if (not is_fake_exgauss):
    x_obs = []
    y_obs = []
    w_obs = []
#    fin = open("/net/cci-filer2/raid1/home/asmit/mcmc/real_data/LD91/merging/CDF_reflection.dat",'r')
    #fin = open("exgauss_simulated_intensities.dat",'r')
    fin = open(sys.argv[1],'r')
#    fin = open("exgauss_simulated_intensities_1000pts_dx1.dat",'r')
    for line in fin:
      miller_index = '0_0_0'
      if '#' in line:
        ax = line.split()
        miller_index = '%s_%s_%s'%(ax[-3],ax[-2],ax[-1])
      else:
        ax = line.split()
        if float(ax[0]) > -1.e15 and float(ax[0]) < 1.e15:
          x_obs.append(float(ax[0]))
#          y_obs.append(float(ax[1]))
          w_obs.append(1.0)
    fin.close()
  else:
    fake_data_params = [-1000.0, 5000.0, 5000.0]
    x_obs = range(-20000, 20000, 10)
    y_obs = exgauss_cdf_nparray(x_obs, fake_data_params[0], fake_data_params[1], fake_data_params[2])
    w_obs = [1.0 for elem in x_obs]

# Gaussian Data
else:
  fake_data_params = [-1000, 5000]
  x_obs = range(-20000, 20000, 10)
  y_obs = gauss_cdf_nparray(x_obs, fake_data_params[0], fake_data_params[1]) + 0.01 * np.random.normal(size=len(x_obs))
  w_obs = [1.0 for elem in x_obs]

# Adjust CDF to make it (n-0.5)/N instead of n/N to create a buffer
x_obs = np.sort(x_obs)
y_obs = np.array(range(1,len(x_obs)+1))/float(len(x_obs))
y_obs[:] = [z-0.5/len(x_obs) for z in y_obs]
# scale the data if it is real data ?
#if (exgauss and not is_fake_exgauss):
#  x_obs = (x_obs - np.min(x_obs))#/(np.max(x_obs)- np.min(x_obs))

x_obs = flex.double(x_obs)
y_obs = flex.double(y_obs)
w_obs = flex.double(w_obs)


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

if (exgauss):
#  initial = flex.double([-4000, 4000, 25000])
  initial = flex.double(initial_guess(x_obs))
else:
  initial = flex.double([-500, 3000])

class lbfgs_gauss():
  def __init__(self, x_obs, y_obs, w_obs, initial):
#    super(lbfgs_exgauss,self).__init__()
#    self.set_cpp_data(x_obs,y_obs,w_obs)
    assert x_obs.size() == y_obs.size()
    self.x_obs = x_obs
    self.y_obs = y_obs
    self.w_obs = w_obs #flex.double([1.0]*(w_obs.size()))
    self.n = len(initial)
    self.x = initial.deep_copy()
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self,
                     termination_params = scitbx.lbfgs.termination_parameters
                     (traditional_convergence_test_eps=1.e-10), core_params = 
                     scitbx.lbfgs.core_parameters(gtol=0.001))
    self.a = self.x

  def gauss_cdf(self,x,mu,sigma):
    return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

  def gauss_cdf_derivatives(self,x,mu,sigma):
    grad = [0., 0.]
    z = (x-mu)/(sigma*np.sqrt(2))
    grad[0] = (-1./(np.sqrt(2*np.pi*sigma*sigma)))*(1.)*np.exp(-z*z)
    grad[1] = (-1./(np.sqrt(2*np.pi*sigma*sigma)))*((x-mu)/(sigma))*np.exp(-z*z)
    return grad    

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0])
    for i in range(self.x_obs.size()):
      y_calc = self.gauss_cdf(self.x_obs[i], self.x[0], self.x[1])
      y_diff = self.y_obs[i] - y_calc
      result += y_diff*y_diff*self.w_obs[i]
      prefactor = -2.*self.w_obs[i]*y_diff
      mu = self.x[0]
      sigma = self.x[1]
      z = (self.x_obs[i]-mu)/(sigma*np.sqrt(2))
      grad[0] += prefactor*(-1./(np.sqrt(2*np.pi*sigma*sigma)))*(1.)*np.exp(-z*z)
      grad[1] += prefactor*(-1./(np.sqrt(2*np.pi*sigma*sigma)))*((self.x_obs[i]-mu)/(sigma))*np.exp(-z*z)
    return result,grad

  def compute_functional_and_gradients(self):
    self.a = self.x
    f,g = self.target_func_and_grad()
    self.print_step("LBFGS EXGAUSS stp",f)
    return f, g

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target),
    print "["," ".join(["%10.4f"%a for a in pfh.x]),"]"


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
                     (traditional_convergence_test_eps=1.e-10), core_params =
                     scitbx.lbfgs.core_parameters(gtol=0.00011))
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
      cdf.append(self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v+np.log(self.gauss_cdf(u,v*v,v))))
    return np.array(cdf)

  def exgauss_cdf(self,x, mu, sigma, tau):
    u = (x-mu)/tau
    v = sigma/tau
#    print u,v,self.gauss_cdf(u,0,v), self.gauss_cdf(u,v*v,v)
    return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v+np.log(self.gauss_cdf(u,v*v,v)))

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0,0])
    for i in range(0, self.x_obs.size()):
      y_calc = self.exgauss_cdf(self.x_obs[i], self.x[0], self.x[1], self.x[2]) 
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
      exp_2 = np.exp(-z2*z2)/np.sqrt(np.pi*phi_uv2v*phi_uv2v)
      exp_0 = np.exp(-u+0.5*v*v+np.log(phi_uv2v))
#      print exp_1, exp_2, exp_0, phi_uv2v
      grad[0] += prefactor*((-exp_1) - exp_0*((1./tau) + (exp_2*(-1./(sigma*np.sqrt(2))))))
      grad[1] += prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma))))
      grad[2] += prefactor*((0.0) - exp_0*((u/tau) - (v*v/tau) + (exp_2*(v/(tau*np.sqrt(2))))))

#    print 'gradients', grad[0], grad[1], grad[2]
    return result,grad

  def compute_functional_and_gradients(self):
    self.a = self.x
    f,g = self.target_func_and_grad()
    self.print_step("LBFGS EXGAUSS stp",f)
    return f, g

def lbfgs_example(verbose):
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
  plt.plot(x_obs,y_obs,'o')#, facecolors='None',edgecolors='b')
#  fout = open('exgauss_simulated.dat','w')
#  for i in range(len(X1)):
#    fout.write("%12.4f" %X1[i])
#    fout.write("%12.4f\n" %y_obs[i])
  if (exgauss):
    F0 = fit.exgauss_cdf_nparray(X1, initial[0], initial[1], initial[2])
    F1 = fit.exgauss_cdf_nparray(X1,fit.a[0],fit.a[1], fit.a[2])
    F2 = fit.exgauss_cdf_nparray(X1,-4000.0,4000.0, 25000.0)
  else:
    F0 = gauss_cdf_nparray(X1, initial[0], initial[1])
    F1 = gauss_cdf_nparray(X1, fit.a[0], fit.a[1])
  plt.plot(X1, F0, 'r*', linewidth=2.0)
  plt.plot(X1,F1,'g+',linewidth=2)
#  plt.plot(X1,F2,'k-',linewidth=2)
  import os
  seed = os.path.split(sys.argv[1])[-1].split('.')[0].split('_')[-1]
  from construct_random_datapt import ExGauss
  EXG= ExGauss(10000, -200000, 200000, fit.a[0], fit.a[1], fit.a[2])
  cdf_cutoff = 0.95
  I_fit = EXG.interpolate_x_value(cdf_cutoff)
  print I_fit
  fout = open('intensity_cdf.dat','a')
  fout.write("%12.5f\n"%I_fit) 
  fout.close()
  plt.plot(X1, [cdf_cutoff]*len(X1), 'r--')
  plt.plot([I_fit]*100, np.linspace(0,1,100),'r--' )
  plt.savefig('fit_intensities_%s.pdf'%seed)
#  plt.figure(2)
#  plt.plot(X1, F2-np.array(y_obs), 'o') 
#  plt.plot(X1, [0.0]*len(X1), 'r--')
#  plt.ylabel('$\Delta(CDF_{Theory}-CDF_{Calc})$', fontsize=18)
#  plt.xlabel('$ Intensity $', fontsize=18)
#  plt.hist(x_obs, normed=True,bins=100)
#  plt.show()

if (__name__ == "__main__"):
  verbose=True
  print "\n LBFGS:"
  lbfgs_example(verbose)
