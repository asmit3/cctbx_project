from construct_random_datapt import ExGauss
import matplotlib.pyplot as plt
import numpy as np
import sys

seed = int(sys.argv[1])
exgauss= ExGauss(20, -200000, 200000, -4000.0, 4000.0, 25000.0)
#exgauss_random_array = p.map(exgauss.rand())
#exgauss_random_array = exgauss.rand(seed)
exgauss_random_array = exgauss.rand_annlib(seed)
print 'created numbers'
X2 = np.sort(exgauss_random_array)
F2 = np.array(range(1,len(exgauss_random_array)+1))/float(len(exgauss_random_array))
# Nick's suggestion on using (n-0.5)/N for cdf to create a buffer region
F2[:] = [x-0.5/len(exgauss_random_array) for x in F2]
fout = open('exgauss_simulated_intensities_'+sys.argv[1]+'.dat', 'w')
for i in range(len(X2)):
  fout.write("%12.7f"%X2[i])
  fout.write("%12.7f\n"%F2[i])
print exgauss.interpolate_x_value(0.95)

#plt.plot(X2,F2,'o')


