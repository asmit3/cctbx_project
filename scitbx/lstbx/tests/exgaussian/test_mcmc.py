import sys
from scitbx.lstbx.tests.exgaussian.mcmc_exgauss import mcmc_exgauss
import numpy as np
from scitbx.array_family import flex
# Try an example data from a file like 10/exgaussian_intensities_22.dat

from memory_profiler import profile

@profile
class test_mcmc():
  def __init__(self):
    pass

  def run(self):
    mcmc_1 = mcmc_exgauss(datasource=sys.argv[1], cdf_cutoff=0.95, nsteps=10, t_start=1, dt=1, plot=False)
    mcmc_1.run()
    del(mcmc_1)
# Try from different data sources by commenting out relevant lines
# Here the data is from 10/exgaussian_intensities_22.dat
    print '############################################################################'
    data=[ 711.0616599,
       2007.6860195,
       2419.3656149,
       4161.9560098, 
       6657.3426264, 
       9960.9449851,
       12749.0884098,
       25683.6957858,
       38096.3072357,
       45327.1748034] 

# a. list
#datasource = data
# b. np.ndarray
#datasource = np.array(data)
# c. flex.double
    datasource = flex.double(data)
# d. flex.int 
#datasource = flex.int(map(int, data)) 
# Now run the mcmc stuff
    mcmc_2 = mcmc_exgauss(datasource, 0.95, 10, 1, 1, plot=False)
    mcmc_2.run()

if __name__ == '__main__':
  import sys
  mcmc_test = test_mcmc()
  mcmc_test.run()
