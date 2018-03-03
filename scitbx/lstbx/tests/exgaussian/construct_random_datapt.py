# Defines functions that act as Ex-gaussian random number generators
import math
import numpy as np
from scitbx.array_family import flex

class ExGauss():
  def __init__(self, N, xmin, xmax, mu, sigma, tau):
    self.N = N
    self.xmin = int(xmin)
    self.xmax = int(xmax)
    self.mu = mu
    self.sigma = sigma
    self.tau = tau
    self.dx = 1000 #int((int(xmax)-int(xmin))/10000)

  def gauss_cdf(self,x, mu, sigma):
    return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

  def exgauss_cdf_nparray(self,data):
    cdf = []
    for x in data:
      x = float(x)
      u = (x-self.mu)/self.tau
      v = self.sigma/self.tau
      if self.gauss_cdf(u,v*v,v) == 0.0:
        cdf.append(self.gauss_cdf(u,0,v))
      else:
        cdf.append(self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v)))
    return np.array(cdf)

  def exgauss_cdf(self,x):
    u = (x-self.mu)/self.tau
    v = self.sigma/self.tau
    if self.gauss_cdf(u,v*v,v) == 0.0:
      return self.gauss_cdf(u,0,v)
    else:
      return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v))
#    return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v+np.log(self.gauss_cdf(u,v*v,v)))
  
  def create_exgauss_lookup_table(self):
    """
      creates an ex-gaussian CDF lookup table that will be used to interpolate values
    """
    return self.exgauss_cdf_nparray(range(self.xmin,self.xmax, self.dx)).tolist(), range(self.xmin,self.xmax, self.dx)


  def create_smarter_lookup_table(self, y=0.95):
    """
      creates an ex-gaussian CDF lookup table but in a smarter way in that it creates a leaner look-up table with
      more values stacked closer to where the cdf value we are searching for is
    """
  # First determine an approximate starting point for the lookup taqble by halving the max value till the point 
  # where the cdf value is less than the cdf value we are looking for
    xold = self.xmax
    xnew = self.xmax
    y_calc = self.exgauss_cdf(xnew)
    while y_calc > y:
      xold = xnew
      xnew = xnew/2.
      y_calc = self.exgauss_cdf(xnew)
    
  # Make sure the interval over which this is being constructed is okay
    npts = 10. # Number of data pts in case the interval xold-xnew is smaller than self.dx
    if xold-xnew < self.dx:
      dx = int((xold-xnew)/npts)
    else: 
      dx = self.dx
  # Now start building the lookup table from the value of x
    return self.exgauss_cdf_nparray(range(int(xnew),int(xold), dx)).tolist(), range(int(xnew),int(xold), dx)

  def interpolate_x_value(self, y):
#    return np.random.normal(1000., 10)
#    if self.N < 20:
    lookup_table_cdf, lookup_table_x = self.create_exgauss_lookup_table()
#    else:
    #lookup_table_cdf, lookup_table_x = self.create_smarter_lookup_table(y)
    lookup_table_cdf = flex.double(lookup_table_cdf)
#    print 'Length of lookup table = ', len(lookup_table_cdf)
#    del(lookup_table_cdf)
#    del(lookup_table_x)
#    return np.random.normal(1000., 10)
    from annlib_ext import AnnAdaptor 
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
#      print 'true cdf value from interpol',self.exgauss_cdf(x)
      return x
    except:
      print 'in the except block', y
#      print 'true cdf value from interpol',self.exgauss_cdf(x)
      return lookup_table_x[idx]

  def find_x_from_iter(self, y):
    # First make an initial guess for x which in turn would be an intial guess for u
    # Assume a straight line connect xmin to xmax
    #
    xold = self.xmax #self.xmin + y*(self.xmax-self.xmin)
    uold = (xold-self.mu)/self.tau  
    v = self.sigma/self.tau
    eps = 1.e-05
    while (self.gauss_cdf(uold,0.,v)-y) < 0.0:
      uold = uold+2.0 
    if self.gauss_cdf(uold, v*v,v) == 0.0:
      unew = v*math.sqrt(2)*self.inverse_erf(2*y-1)
    else:
      unew = (v*v/2.0)+np.log(1.0*self.gauss_cdf(uold, v*v,v)/(self.gauss_cdf(uold,0.,v)-y))
#      print 'who new',unew,v, self.gauss_cdf(uold, v*v,v), self.gauss_cdf(uold,0.,v)-y 
#    from IPython import embed; embed(); exit()
    count = 0
    while np.abs((unew-uold)/uold) > eps:
      count +=1
      uold=unew
      if self.gauss_cdf(uold, v*v,v) == 0.0:
        unew = v*math.sqrt(2)*self.inverse_erf(2*y-1)
      else:
        unew = (v*v/2.0)+np.log(1.0*self.gauss_cdf(uold, v*v,v)/(self.gauss_cdf(uold,0.,v)-y))
#    try:
#      unew = np.log(np.exp(v*v/2)*self.gauss_cdf(uold, v*v,v)/(self.gauss_cdf(uold,0.,v)-y))
#    except:
#      unew = v*math.sqrt(2)*self.inverse_erf(2*y-1)
#      unew = np.log(np.exp(v*v/2)*self.gauss_cdf(uold, v*v,v)/(self.gauss_cdf(uold,0.,v)-y))
#    print 'true cdf value from iter and niter = ',self.exgauss_cdf(unew*self.tau+self.mu),count
#    if np.isnan(unew):
#      from IPython import embed; embed();exit()      
    return unew*self.tau+self.mu

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
    from annlib_ext import AnnAdaptor 
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

  def inverse_erf(self, z):
#    if z > 1.0 or z < 0.0
#      raise Exception('z value cannot be inverted by inverse_erf')
    maclaurin_sum = 0.0
    nterms = 25
    c = [1.0]
    # First obtain the coefficients
    for k in xrange(1,nterms):
      c_tmp = 0.0
      for m in xrange(0,k):
        c_tmp += c[m]*c[k-1-m]/((m+1)*(2*m+1))
      c.append(c_tmp)
    for k in range(0,nterms):
      maclaurin_sum += (c[k]/(2*k+1))*(math.pow((math.sqrt(math.pi)*z/2.0),2*k+1))
    return maclaurin_sum

