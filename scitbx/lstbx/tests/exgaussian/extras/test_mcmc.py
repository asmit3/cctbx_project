import sys
from mcmc_exgauss import mcmc_exgauss
mcmc_test = mcmc_exgauss(sys.argv[1], 0.95, 10, 1, 1, plot=True)
mcmc_test.run()
