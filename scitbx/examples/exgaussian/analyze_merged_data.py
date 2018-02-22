#
import numpy as np 
from libtbx import easy_pickle
#from test3_logarithm import *

def exgauss_pdf(data, mu,sigma, tau):
    y = []
    for x in data:
        y.append(np.exp((sigma*sigma-2.0*tau*(x-mu))/(2.0*tau*tau))*(1e-15+1.0-math.erf((sigma*sigma-tau*(x-mu))/(sigma*tau*math.sqrt(2)))))
    return np.array(y)

data = easy_pickle.load('merged_images.pickle')

reflections = len(data.keys())
Scaled_Intensity = []
# play around with index 0
#for datapt in data[data.keys()[0]]:
#    Scaled_Intensity.append(datapt[0])

# Print out all the reflections in separate files
for reflection in data.keys():
    fout = open('reflections/intensities_%s_%s_%s.dat'%(reflection[0],reflection[1],reflection[2]),'w')
    fout.write('# Miller Index %s %s %s\n'%(reflection[0],reflection[1],reflection[2]))
    for datapt in data[reflection]:
        fout.write("%12.8f\n"%datapt[0]) 
    fout.close()
exit()
print len(Scaled_Intensity)
total_instances = len(Scaled_Intensity)

# Print out this Scaled intensity and exit for now
fout = open('reflection.dat','w')
for entry in Scaled_Intensity:
    fout.write("%12.3f\n" %entry)
exit()

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#n, bins, patches = plt.hist(Scaled_Intensity, 10,facecolor='green', alpha=0.75)
#plt.xlabel('Intensity')
#plt.ylabel('Frequency')
#plt.title('Intensity distribution for miller indices %s' %str(data.keys()[0]))

Scaled_Intensity = np.array(Scaled_Intensity)
mu = np.mean(Scaled_Intensity)-skewness(Scaled_Intensity)
tau = np.std(Scaled_Intensity)*0.8
sigma = np.sqrt(np.var(Scaled_Intensity)-tau*tau)
nsteps = 50000
maxI = np.max(Scaled_Intensity)
minI = np.min(Scaled_Intensity)
print 'initial guesses', mu,sigma,tau
params = sampler(Scaled_Intensity, samples=nsteps, mu_init= mu,sigma_init = sigma,tau_init = tau,proposal_width = 0.001*np.abs(maxI-minI),plot=False)

mu,sigma, tau = params[-1]
print 'final parameter values ',mu,sigma, tau
X1 = np.arange(min(Scaled_Intensity), max(Scaled_Intensity),100.0)
for count in range(40000, nsteps,10000):
#    x = np.arange(minI, maxI, 100.0)
    mu,sigma, tau = params[count]
#    y = total_instances*exgauss_pdf(x, mu, sigma, tau)
#    l = plt.plot(x,y, 'grey', linewidth=0.3)
# Get CDFs
#X1 = np.arange(min(Scaled_Intensity), max(Scaled_Intensity),10.0)
    F1 = exgauss_cdf(X1,mu,sigma,tau)
    plt.plot(X1,F1,'grey',linewidth=0.3)
X2 = np.sort(Scaled_Intensity)
F2 = np.array(range(len(Scaled_Intensity)))/float(len(Scaled_Intensity))
plt.plot(X2,F2,'o')


#plt.plot(Scaled_Intensity)
#plt.ylabel('Intensity')
plt.show()
