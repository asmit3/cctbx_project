# This script is for pinting out the mean(I_fit) and sigma(I_fit) from the test values generated
import numpy as np
import matplotlib.pyplot as plt

plot=True
fin = open('intensity_cdf.dat','r')
intensity = []
for line in fin:
  if line !='\n':
    intensity.append(float(line.split()[0]))

print 'Mean of 95 percentile intensity value = %12.5f'%np.mean(intensity)
print 'Stdev of 95 percentile intensity value = %12.5f'%np.std(intensity)

if (plot):
  plt.hist(intensity,normed=True, bins=20)
  plt.show()
