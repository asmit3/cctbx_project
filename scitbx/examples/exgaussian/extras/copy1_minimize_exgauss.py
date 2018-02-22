from __future__ import division
import scitbx.lbfgs
from scitbx.array_family import flex
from math import exp,sqrt
from scitbx.matrix import sqr

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
  

#raw_strings = data.split("\n")
#data_index = raw_strings.index("Time Counts")
#x_obs = flex.double([float(line.split()[0]) for line in raw_strings[data_index+1:data_index+60]])
#y_obs = flex.double([float(line.split()[1]) for line in raw_strings[data_index+1:data_index+60]])
#w_obs = 1./(y_obs)
initial = flex.double([24477, 131533, 175377])
#initial = flex.double([10.4, 958.3, 131.4, 33.9, 205])

from scitbx.examples.bevington import bevington_silver
class lbfgs_biexponential_fit (bevington_silver):
  def __init__(self, x_obs, y_obs, w_obs, initial):
    super(lbfgs_biexponential_fit,self).__init__()
    self.set_cpp_data(x_obs,y_obs,w_obs)
    assert x_obs.size() == y_obs.size()
    self.x_obs = x_obs
    self.y_obs = y_obs
    self.w_obs = w_obs
    self.n = len(initial)
    self.x = initial.deep_copy()
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self)
    self.a = self.x

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target),
    print "["," ".join(["%10.4f"%a for a in pfh.x]),"]"

  def compute_functional_and_gradients(self):
    self.a = self.x
    f = self.functional(self.x)
    self.print_step("LBFGS stp",f)
    g = self.gvec_callable(self.x)
    return f, g


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

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target),
    print "["," ".join(["%10.4f"%a for a in pfh.x]),"]"

  def curvatures(self):
    return flex.double([1, 1, 1, 1, 1])

# Good blog on different minimizers
# http://aria42.com/blog/2014/12/understanding-lbfgs

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0,0,0,0])
    for i in range(x_obs.size()):
      y_calc = self.x[0] + self.x[1]*np.exp(-self.x_obs[i]/self.x[3])+self.x[2]*np.exp(-self.x_obs[i]/self.x[4])
      y_diff = y_obs[i] - y_calc
      result += y_diff*y_diff*self.w_obs[i]
      prefactor = -2.*w_obs[i]*y_diff
      grad[0] += prefactor
      grad[1] += prefactor*np.exp(-self.x_obs[i]/self.x[3])
      grad[2] += prefactor*np.exp(-self.x_obs[i]/self.x[4])
      grad[3] += prefactor*self.x[1]*np.exp(-self.x_obs[i]/self.x[3])*(self.x_obs[i]/(self.x[3]*self.x[3]))
      grad[4] += prefactor*self.x[2]*np.exp(-self.x_obs[i]/self.x[4])*(self.x_obs[i]/(self.x[4]*self.x[4]))
    return result,grad

  def compute_functional_and_gradients(self):
    self.a = self.x
    f,g = self.target_func_and_grad()
    self.print_step("LBFGS EXGAUSS stp",f)
    return f, g

class lbfgs_biexpfit ():
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

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target),
    print "["," ".join(["%10.4f"%a for a in pfh.x]),"]"

  def curvatures(self):
    return flex.double([1, 1, 1, 1, 1])

# Good blog on different minimizers
# http://aria42.com/blog/2014/12/understanding-lbfgs

  def target_func_and_grad(self):
    import numpy as np
    result = 0.0
    grad = flex.double([0,0,0,0,0])
    for i in range(x_obs.size()):
      y_calc = self.x[0] + self.x[1]*np.exp(-self.x_obs[i]/self.x[3])+self.x[2]*np.exp(-self.x_obs[i]/self.x[4])
      y_diff = y_obs[i] - y_calc
      result += y_diff*y_diff*self.w_obs[i]
      prefactor = -2.*w_obs[i]*y_diff
      grad[0] += prefactor
      grad[1] += prefactor*np.exp(-self.x_obs[i]/self.x[3])
      grad[2] += prefactor*np.exp(-self.x_obs[i]/self.x[4])
      grad[3] += prefactor*self.x[1]*np.exp(-self.x_obs[i]/self.x[3])*(self.x_obs[i]/(self.x[3]*self.x[3]))
      grad[4] += prefactor*self.x[2]*np.exp(-self.x_obs[i]/self.x[4])*(self.x_obs[i]/(self.x[4]*self.x[4]))
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
