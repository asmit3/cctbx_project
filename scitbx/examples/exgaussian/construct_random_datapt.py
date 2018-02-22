# Defines functions that act as Ex-gaussian random number generators
import math
import numpy as np
from annlib_ext import AnnAdaptor 
from scitbx.array_family import flex
import time

class ExGauss():
  def __init__(self, N, xmin, xmax, mu, sigma, tau):
    self.N = N
    self.xmin = int(xmin)
    self.xmax = int(xmax)
    self.mu = mu
    self.sigma = sigma
    self.tau = tau
    self.dx = 1000

  def gauss_cdf(self,x, mu, sigma):
    return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

  def exgauss_cdf_nparray(self,data):
    cdf = []
    for x in data:
      x = float(x)
      u = (x-self.mu)/self.tau
      v = self.sigma/self.tau
      cdf.append(self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v+np.log(self.gauss_cdf(u,v*v,v)+1e-15)))
    return np.array(cdf)

  def exgauss_cdf(self,x):
    u = (x-self.mu)/self.tau
    v = self.sigma/self.tau
    return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v+np.log(self.gauss_cdf(u,v*v,v)))
  
  def create_exgauss_lookup_table(self):
    """
      creates an ex-gaussian CDF lookup table that will be used to interpolate values
    """
    return self.exgauss_cdf_nparray(range(self.xmin,self.xmax, self.dx)).tolist(), range(self.xmin,self.xmax, self.dx)

  def interpolate_x_value(self, y):
    lookup_table_cdf, lookup_table_x = self.create_exgauss_lookup_table()
    lookup_table_cdf = flex.double(lookup_table_cdf)
    A = AnnAdaptor(lookup_table_cdf, 1)
    A.query([y])
    idx = A.nn[0]
    try:
      y1 = lookup_table_cdf[idx]
      x1 = lookup_table_x[idx]
      if y > y1:
        y2 = lookup_table_cdf[idx+1]
        x2 = lookup_table_x[idx+1]
      else:
        y2 = lookup_table_cdf[idx-1]
        x2 = lookup_table_x[idx-1]
      x = ((x2-x1)/(y2-y1))*(y-y1) + x1
      return x
    except:
      print 'in the except block', y
      return lookup_table_x[idx]

# See https://www.av8n.com/physics/arbitrary-probability.htm 
  def rand_naive(self, seed):
    exgauss_rand = []
    np.random.seed(seed)
    unirand = np.random.rand(self.N)
    lookup_table_cdf, lookup_table_x = self.create_exgauss_lookup_table()
    for y in unirand:
      idx = lookup_table_cdf.index(min(lookup_table_cdf, key=lambda y0:abs(y0-y)))
      try:
        y1 = lookup_table_cdf[idx]
        x1 = lookup_table_x[idx]
        if y > y1:
          y2 = lookup_table_cdf[idx+1]
          x2 = lookup_table_x[idx+1]
        else:
          y2 = lookup_table_cdf[idx-1]
          x2 = lookup_table_x[idx-1]
        x = ((x2-x1)/(y2-y1))*(y-y1) + x1
        exgauss_rand.append(x)
      except:
        print 'in the except block', y
        exgauss_rand.append(lookup_table_x[lookup_table_cdf.index(min(lookup_table_cdf, key=lambda y0:abs(y0-y)))])
#      exgauss_rand.append(lookup_table_x[lookup_table_cdf.index(min(lookup_table_cdf, key=lambda x:abs(x-number)))])
    return exgauss_rand        

  def rand_annlib(self, seed):
    exgauss_rand = []
    np.random.seed(seed)
    query = np.random.rand(self.N)
    lookup_table_cdf, lookup_table_x = self.create_exgauss_lookup_table()
    lookup_table_cdf = flex.double(lookup_table_cdf)
#    t1 = time.time()
    A = AnnAdaptor(lookup_table_cdf, 1)
    A.query(query)
# ================== If you want to test timing comment out below ================
#    for i in xrange(len(A.nn)):
#      print "Neighbor of (%12.7f), index %6d distance %12.5f"%(
#      query[i],A.nn[i],math.sqrt(A.distances[i]))
#    t2 = time.time()
#    print 'Time Taken by Annlib = %12.7f'%(t2-t1)
#    lookup_table_cdf = lookup_table_cdf.as_numpy_array().tolist()
#    t1 = time.time()
#    for y in query:
#      idx = lookup_table_cdf.index(min(lookup_table_cdf, key=lambda y0:abs(y0-y)))
#      print 'From Naive Search (%12.7f), index %6d'%(y, idx)
#    t2 = time.time()
#    print 'Time taken by Naive = %12.7f'%(t2-t1)
## ====================== Test Over ================================================
    for i in xrange(len(A.nn)):
      idx = A.nn[i]
      y = query[i]
      try:
        y1 = lookup_table_cdf[idx]
        x1 = lookup_table_x[idx]
        if y > y1:
          y2 = lookup_table_cdf[idx+1]
          x2 = lookup_table_x[idx+1]
        else:
          y2 = lookup_table_cdf[idx-1]
          x2 = lookup_table_x[idx-1]
        x = ((x2-x1)/(y2-y1))*(y-y1) + x1
        exgauss_rand.append(x)
      except:
        print 'in the except block'
        exgauss_rand.append(lookup_table_x[idx])
    return exgauss_rand 

