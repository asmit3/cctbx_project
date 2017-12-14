from __future__ import division
from scitbx.array_family import flex
from scitbx import sparse
from scitbx.lstbx import normal_eqns, normal_eqns_solving
from libtbx.test_utils import approx_equal, Exception_expected

class exponential_fit(
  normal_eqns.non_linear_ls,
  normal_eqns.non_linear_ls_mixin):

  """ Model M(x, t) = x3 e^{x1 t} + x4 e^{x2 t}

      Problem 18 from

      UCTP Test Problems for Unconstrained Optimization
      Hans Bruun Nielsen
      TECHNICAL REPORT IMM-REP-2000-17
  """

  n_data = 45
  arg_min = flex.double((-4, -5, 4, -4))
  x_0     = flex.double((-1, -2, 1, -1))

  def __init__(self):
    super(exponential_fit, self).__init__(n_parameters=4)
    self.t = 0.02*flex.double_range(1, self.n_data + 1)
    self.y = flex.double((
      0.090542, 0.124569, 0.179367, 0.195654, 0.269707, 0.286027, 0.289892,
      0.317475, 0.308191, 0.336995, 0.348371, 0.321337, 0.299423, 0.338972,
      0.304763, 0.288903, 0.300820, 0.303974, 0.283987, 0.262078, 0.281593,
      0.267531, 0.218926, 0.225572, 0.200594, 0.197375, 0.182440, 0.183892,
      0.152285, 0.174028, 0.150874, 0.126220, 0.126266, 0.106384, 0.118923,
      0.091868, 0.128926, 0.119273, 0.115997, 0.105831, 0.075261, 0.068387,
      0.090823, 0.085205, 0.067203
      ))
    assert len(self.y) == len(self.t)
    self.restart()

  def restart(self):
    self.x = self.x_0.deep_copy()
    self.old_x = None

  def parameter_vector_norm(self):
    return self.x.norm()

  def build_up(self, objective_only=False):
    x1, x2, x3, x4 = self.x
    exp_x1_t = flex.exp(x1*self.t)
    exp_x2_t = flex.exp(x2*self.t)
    residuals = x3*exp_x1_t + x4*exp_x2_t
    residuals -= self.y

    self.reset()
    if objective_only:
      self.add_residuals(residuals, weights=None)
    else:
      grad_r = (self.t*x3*exp_x1_t,
                self.t*x4*exp_x2_t,
                exp_x1_t,
                exp_x2_t)
      jacobian = flex.double(flex.grid(self.n_data, self.n_parameters))
      for j, der_r in enumerate(grad_r):
        jacobian.matrix_paste_column_in_place(der_r, j)
      self.add_equations(residuals, jacobian, weights=None)

  def step_forward(self):
    self.old_x = self.x.deep_copy()
    self.x += self.step()

  def step_backward(self):
    assert self.old_x is not None
    self.x, self.old_x = self.old_x, None

def exercise_levenberg_marquardt(non_linear_ls):
  non_linear_ls.restart()
  iterations = normal_eqns_solving.levenberg_marquardt_iterations(
    non_linear_ls,
    track_all=True,
    gradient_threshold=1e-8,
    step_threshold=1e-8,
    tau=1e-4,
    n_max_iterations=200)
  assert non_linear_ls.n_equations == non_linear_ls.n_data
  assert approx_equal(non_linear_ls.x, non_linear_ls.arg_min, eps=5e-4)
  print "L-M: %i iterations" % iterations.n_iterations

def run():
  import sys
  t = exponential_fit()
  exercise_levenberg_marquardt(t)
  print 'OK'

if __name__ == '__main__':
  run()
