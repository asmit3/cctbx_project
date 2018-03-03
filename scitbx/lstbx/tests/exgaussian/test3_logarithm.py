import numpy as np
import matplotlib.pyplot as plt
import math
from scitbx.array_family import flex
from construct_random_datapt import ExGauss
# http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
# https://github.com/mwaskom/seaborn/issues/351 for seaborn issues
# Now try varying sigma and mu !!
# Implemented using logarithm for simplicity

class mcmc():
  def __init__(self):
    self.proposal_factor = 2.0

  def gauss_pdf(self,x,mu,sigma):
    return np.log(1./(np.sqrt(2.0*np.pi*sigma*sigma))) - (((x-mu)*(x-mu))/(2.0*sigma*sigma))

  def exgauss(self,data,mu, sigma, tau):
    sumofprob = 0.0
    ct = 0
    for x in data:
      if math.erfc((sigma*sigma-tau*(x-mu))/(sigma*tau*math.sqrt(2))) == 0.0 :
        return -np.inf 
      tmp_term= np.log(1./(2*tau))+((sigma*sigma-2.0*tau*(x-mu))/(2.0*tau*tau)) + np.log(math.erfc((sigma*sigma-tau*(x-mu))/(sigma*tau*math.sqrt(2))))
      sumofprob += tmp_term
    return sumofprob

  def gauss_cdf(self,x,mu,sigma):
    return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

  def exgauss_cdf_array(self,data,mu,sigma,tau):
    cdf = []
    for x in data:
      u = (x-mu)/tau
      v = sigma/tau
      if self.gauss_cdf(u,v*v,v) == 0.0:
        cdf.append(self.gauss_cdf(u,0,v))
      else:
        cdf.append(self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v)))
    return flex.double(cdf)

  def exgauss_cdf(self,x, mu, sigma, tau):
    u = (x-mu)/tau
    v = sigma/tau
    if self.gauss_cdf(u,v*v,v) == 0.0:
      return self.gauss_cdf(u,0,v)
    else:
      return self.gauss_cdf(u,0,v)-np.exp(-u+0.5*v*v)*(self.gauss_cdf(u,v*v,v))
  
  def gamma_dist_pdf(self, x, k, theta):
    from scitbx.math import gamma_complete
    return (k-1)*np.log(x)-np.log(gamma_complete(k))-k*np.log(theta)-(x/theta)

  def sampler(self, data, samples=5, mu_init=0.5, sigma_init = 0.5,tau_init = 0.5, 
              proposal_width = 10.0,t_start=1, dt=1,cdf_cutoff=0.95, 
              plot=False,analyse_mcmc = False,seed=22, residual = 0.0, prior_errors = [1., 1., 1.]):
    np.random.seed(seed) # was with seedhere
    data = np.sort(data)
    maxI = np.max(data) #+np.max(data)/2.
    minI = np.min(data) #-np.min(data)/2.
    mu_current = mu_init
    sigma_current = sigma_init
    tau_current = tau_init
#    print 'PRIOR ERROS from BOOTSTRAP TECHNIQUE= ',prior_errors
#    dmu,dsigma,dtau = prior_errors 
    dmu,dsigma,dtau = self.get_error_estimate_of_params(residual, data, mu_init, sigma_init, tau_init)
#    if np.abs(dmu) > 1e6 or np.abs(dsigma) > 1e6 or np.abs(dtau) > 1e6 or np.abs(dmu) < 1.0 or np.abs(dsigma) < 1.0 or np.abs(dtau) < 1.0:
#      dmu,dsigma,dtau = prior_errors 
#      print 'PRIOR ERROS from BOOTSTRAP TECHNIQUE= ',prior_errors
    
    mu_prior_mu = mu_init
    mu_prior_sd = dmu #(np.abs(mu_alt-mu_init))/2000.
    sd_prior_mu = sigma_init
    sd_prior_sd = dsigma #(np.abs(sigma_alt-sigma_init))/2000.
    tau_prior_mu = tau_init
    tau_prior_sd = dtau #np.abs(tau_alt-tau_init)/2000.
    accept_counter = 0
#
#    self.proposal_factor = self.find_optimal_proposal_factor()
    posterior = [[mu_current,sigma_current, tau_current]]
    I_mcmc = []
    for i in range(samples):
      accept = False
      inf_flag = False
# trial move
      mu_proposal =  np.random.normal(mu_current,self.proposal_factor* mu_prior_sd/1.)
      sigma_proposal = np.abs(np.random.normal(sigma_current, self.proposal_factor*sd_prior_sd/1.)) #
      tau_proposal = np.abs(np.random.normal(tau_current, self.proposal_factor*tau_prior_sd/1.))

      likelihood_current = self.exgauss(data,mu_current, sigma_current, tau_current) # multiply the probabilties
      likelihood_proposal = self.exgauss(data,mu_proposal,sigma_proposal, tau_proposal) # multiply

      prior_current = self.gauss_pdf(mu_current, mu_prior_mu, mu_prior_sd)+self.gauss_pdf(sigma_current, sd_prior_mu, sd_prior_sd)+ \
                      self.gauss_pdf(tau_current, tau_prior_mu, tau_prior_sd)
      prior_proposal = self.gauss_pdf(mu_proposal, mu_prior_mu, mu_prior_sd)+self.gauss_pdf(sigma_proposal, sd_prior_mu, sd_prior_sd)+ \
                       self.gauss_pdf(tau_proposal, tau_prior_mu, tau_prior_sd)

      if likelihood_proposal == -np.inf:
        accept = False
        inf_flag = True
      if not accept and not inf_flag:
        p_current = likelihood_current + prior_current
        p_proposal = likelihood_proposal + prior_proposal
        p_accept = p_proposal- p_current
        accept = np.log(np.random.rand()) < p_accept
      if plot:
        plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)
      if accept:
        mu_current = mu_proposal
        sigma_current = sigma_proposal
        tau_current = tau_proposal
        accept_counter +=1
      if i > t_start and i%dt == 0:
        EXG= ExGauss(10000, minI, maxI, mu_current,sigma_current,tau_current)
        I_ideal_temp = EXG.find_x_from_iter(cdf_cutoff)
        I_mcmc.append(I_ideal_temp)
        del(EXG)
        if analyse_mcmc:
          posterior.append([mu_current, sigma_current, tau_current])
    print 'acceptance rate',accept_counter*1.0/samples
    I_mcmc_avg =  np.mean(I_mcmc)
    I_mcmc_var =  np.var(I_mcmc)
    if analyse_mcmc:
      self.mcmc_statistics(posterior,I_mcmc,data, cdf_cutoff)
    del(I_mcmc)
    return I_mcmc_avg, I_mcmc_var, accept_counter*1.0/samples

  def mcmc_statistics(self, posterior, I_mcmc, data, cdf_cutoff):
    ''' Get statistics from the ensemble of curves generated using MCMC
        The following statistics are going to be calculated
        a. I_95_avg
        b. I_95_avg - I_95_exp
        c. P(I_95_mcmc) vs delta(I_95_mcmc-I_95_exp)
        d. I_95_mcmc vs t
        e. RMSD(CDF) vs t
        f. RMSF(CDF) vs i (data point)
      Note that here 95 stands for 95 percentile. In effect it is the cdf cutoff value used my mcmc.sampler.
      So don't take the variable name too literally
    '''
    print 'Printing MCMC summary statistics'
    rmsd_flag =False
    # a.
    I_95_avg = np.mean(I_mcmc)
    print 'I_95_avg = ',I_95_avg,'+/-',np.std(I_mcmc)
    # b. 
    I_95_exp, cdf_exp = self.find_x_from_expdata_annlib(data,cdf_cutoff)
    d_I_95_avg = I_95_avg - I_95_exp
    print 'Experimental I_95 = ',I_95_exp
    print 'Deviation of average I_95 from experimental I_95 = ',d_I_95_avg
    #exit()

    plt.figure(3)
    data = np.sort(data)
    fake_pts = np.linspace(np.min(data), np.max(data), 100)
    F1 = np.array(range(1,len(data)+1))/float(len(data))
    F1[:] = [z-0.5/len(F1) for z in F1] # Actual Data
    #F1 = self.exgauss_cdf_array(data,posterior[0][0], posterior[0][1], posterior[0][2]) # Best fit from minimization
    plt.plot(data, F1, '.', linewidth=3.0)
    for count in range(1,len(posterior)):
      F1 = self.exgauss_cdf_array(fake_pts,posterior[count][0], posterior[count][1], posterior[count][2])
      plt.plot(fake_pts, F1, 'grey')
      EXG = ExGauss(10000,np.min(data),np.max(data),posterior[count][0], posterior[count][1], posterior[count][2])
      I_95 = EXG.find_x_from_iter(cdf_cutoff)
      plt.plot(I_95,0.05,'r*')

    F1 = self.exgauss_cdf_array(data,posterior[0][0], posterior[0][1], posterior[0][2])
    plt.plot(data, F1, 'g+', linewidth=3.0)
    # c. 
    d_I_95_mcmc = [np.abs(x - I_95_exp) for x in I_mcmc] 
    print 'Average absolute deviation I_95 mcmc from experimental I_95', np.mean(d_I_95_mcmc)
    hist, bin_edges = np.histogram(d_I_95_mcmc, density=True)
    plt.figure(4)
#    plt.hist(d_I_95_mcmc)
    plt.plot(bin_edges[:-1],hist*np.diff(bin_edges),'-*r')
    plt.xlabel('$\Delta(I_{95_calc}-I_{95_obs})$', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    # d. 
    plt.figure(5)
    plt.plot(range(len(I_mcmc)), I_mcmc, 'b')
    plt.plot(range(len(I_mcmc)), [I_95_exp]*len(I_mcmc), '-*r')
    plt.xlabel('time',fontsize=18)
    plt.ylabel('I_95_mcmc', fontsize=18)
    # e. 
    if (rmsd_flag):
      rmsd = self.calc_rmsd(posterior, data,cdf_exp)
      print 'Average RMSD of datapoints', np.mean(rmsd)
      plt.figure(6)
      plt.plot(range(len(rmsd)), rmsd,'-*r') 
      plt.xlabel('time',fontsize=18)
      plt.ylabel('RMSD',fontsize=18)

    plt.show()

  def calc_rmsd(self, posterior, data,cdf_exp):
    ''' calculates RMSD according to the equation
      rmsd = sum[(x-x0)*(x-x0)]
      Note that no 1/N factor included in denominator, 
      so effectively this is a residual
    '''
    rmsd = []
    for t in range(len(posterior)):
        cdf_t = self.exgauss_cdf_array(data,posterior[t][0],posterior[t][1],posterior[t][2])
#        rmsd.append(np.sqrt(np.sum([(x-y)*(x-y)/1.0 for x,y in zip(cdf_t,cdf_exp)]))) 
        rmsd.append(sum(map(lambda x:x*x,cdf_t-cdf_exp)))
    print rmsd[0], rmsd[1], rmsd[2],rmsd[3]
    return rmsd 
  

  def find_x_from_expdata_annlib(self, data,y):
    ''' find the corresponding x value of a desired y_cdf value, given exp data
    '''
    exgauss_rand = []
    lookup_table_x = np.sort(data)
    lookup_table_cdf = np.array(range(1,len(lookup_table_x)+1))/float(len(lookup_table_x))
    lookup_table_cdf[:] = [z-0.5/len(lookup_table_cdf) for z in lookup_table_cdf]
    y_cdf = lookup_table_cdf
    lookup_table_cdf = flex.double(lookup_table_cdf)
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
      return (x,y_cdf)
    except:
      print 'in the except block of find_x_from_expdata_annlib', y
#      print 'true cdf value from interpol',self.exgauss_cdf(x)
      return (lookup_table_x[idx],y_cdf)

  def get_error_estimate_of_params(self, residual, data, mu0, sigma0,tau0):
    mu2grad,sigma2grad,tau2grad = self.get_residual2_grad(data, mu0, sigma0, tau0) 
    #mu2grad,sigma2grad,tau2grad = self.get_hessian(data, mu0, sigma0, tau0) 
    assert sigma0 > 0., "Initial sigma from minimization is negative, can't assign priors"
    assert tau0 > 0., "Initial tau from minimization is negative, can't assign priors"
    dmu,dsigma,dtau=np.abs(mu0),sigma0,tau0 # error ~ O(value of params) as first guess
    if mu2grad != 0.:
      dmu = np.sqrt(1.0*residual/mu2grad/1.)
    if sigma2grad != 0.:
      dsigma = min(dsigma, np.sqrt(1.0*residual/sigma2grad/1.))
#      else:
#        dsigma = np.sqrt(residual/sigma2grad)
    if tau2grad != 0.:
      dtau = min(dtau,np.sqrt(1.0*residual/tau2grad/1.))
#      else:
#        dtau = np.sqrt(residual/tau2grad)

    print 'my errors = ',dmu,dsigma,dtau
    return dmu,dsigma,dtau

  def get_residual2_grad(self, data, mu, sigma, tau):
    result = 0.0
    mu_grad = 0.0
    sigma_grad = 0.0
    tau_grad = 0.0
    prefactor = 1.0
#    print 'INSIDE RESIDUAL',mu,sigma,tau
    for i in range(0, len(data)):
      y_calc = self.exgauss_cdf(data[i], mu, sigma, tau)
      u = (data[i]-mu)/tau
      v = sigma/tau
      phi_uv2v = self.gauss_cdf(u, v*v, v)
      z1 = (data[i] - mu)/(sigma*np.sqrt(2))
      z2 = (data[i]- mu-(sigma*sigma/tau))/(sigma*np.sqrt(2))
      exp_1 = np.exp(-z1*z1)/np.sqrt(2*np.pi*sigma*sigma)
      exp_2 = np.exp(-z2*z2)/np.sqrt(np.pi)
      exp_0 = np.exp(-u+0.5*v*v)
      if np.isinf(exp_0):
        mu_grad+=(prefactor*((-exp_1)))*(prefactor*((-exp_1)))
        sigma_grad+=(prefactor*((-exp_1*z1*np.sqrt(2))))*(prefactor*((-exp_1*z1*np.sqrt(2))))
        tau_grad+=(prefactor*((0.0)))*(prefactor*((0.0)))
#        print 'isinf mu_grad',mu_grad, (prefactor*((-exp_1)))
      else:
        mu_grad+=(prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2)))))))*(prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2)))))))
        sigma_grad+=(prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((phi_uv2v*sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma)))))*(prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((phi_uv2v*sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma)))))
        tau_grad+=(prefactor*((0.0) - exp_0*((phi_uv2v*u/tau) - (phi_uv2v*v*v/tau) + (exp_2*(v/(tau*np.sqrt(2)))))))*(prefactor*((0.0) - exp_0*((phi_uv2v*u/tau) - (phi_uv2v*v*v/tau) + (exp_2*(v/(tau*np.sqrt(2)))))))
#        print 'is NOT inf mu_grad',mu_grad, (prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2)))))))
    #from IPython import embed; embed(); exit()

    return 1.*mu_grad,1.*sigma_grad,1.*tau_grad

  def get_hessian(self, data, mu, sigma, tau):
    result = 0.0
    mu_grad = 0.0
    sigma_grad = 0.0
    tau_grad = 0.0
    prefactor = 1.0
    lookup_table_x = np.sort(data)
    lookup_table_cdf = np.array(range(1,len(lookup_table_x)+1))/float(len(lookup_table_x))
    lookup_table_cdf[:] = [z-0.5/len(lookup_table_cdf) for z in lookup_table_cdf]
#    print 'INSIDE RESIDUAL',mu,sigma,tau
    for i in range(0, len(data)):
      y_calc = self.exgauss_cdf(data[i], mu, sigma, tau)
      u = (data[i]-mu)/tau
      v = sigma/tau
      phi_uv2v = self.gauss_cdf(u, v*v, v)
      z1 = (data[i] - mu)/(sigma*np.sqrt(2))
      z2 = (data[i]- mu-(sigma*sigma/tau))/(sigma*np.sqrt(2))
      exp_1 = np.exp(-z1*z1)/np.sqrt(2*np.pi*sigma*sigma)
      exp_2 = np.exp(-z2*z2)/np.sqrt(np.pi)
      exp_0 = np.exp(-u+0.5*v*v)    
      ri = lookup_table_cdf[i] - y_calc
      if np.isinf(exp_0):
        # (df/da)*(df/da)
        mu_grad+=(prefactor*((-exp_1)))*(prefactor*((-exp_1)))
        sigma_grad+=(prefactor*((-exp_1*z1*np.sqrt(2))))*(prefactor*((-exp_1*z1*np.sqrt(2))))
        tau_grad+=(prefactor*((0.0)))*(prefactor*((0.0)))
        # -ri*(d2f/da2)
        mu_grad += -ri*((-exp_1*z1*np.sqrt(2)/sigma))
        sigma_grad += -ri*(exp_1*z1*np.sqrt(2)*(1-2*z1*z1)/sigma)
        tau_grad += -ri*(0.0)
      else:
        # (df/da)*(df/da)
        mu_grad+=(prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2)))))))*(prefactor*((-exp_1) - exp_0*((phi_uv2v/tau) + (exp_2*(-1./(sigma*np.sqrt(2)))))))
        sigma_grad+=(prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((phi_uv2v*sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma)))))*(prefactor*((-exp_1*z1*np.sqrt(2)) - exp_0*((phi_uv2v*sigma/(tau*tau))+(exp_2*(-np.sqrt(2)/tau - z2/sigma)))))
        tau_grad+=(prefactor*((0.0) - exp_0*((phi_uv2v*u/tau) - (phi_uv2v*v*v/tau) + (exp_2*(v/(tau*np.sqrt(2)))))))*(prefactor*((0.0) - exp_0*((phi_uv2v*u/tau) - (phi_uv2v*v*v/tau) + (exp_2*(v/(tau*np.sqrt(2)))))))
        # -ri*(d2f/da2)
        mu_grad += -ri*((-exp_1*z1*np.sqrt(2)/sigma))
        sigma_grad += -ri*((exp_1*z1*np.sqrt(2)*(1-2*z1*z1)/sigma))
        tau_grad += -ri*((0.0))
    return mu_grad,sigma_grad,tau_grad


# function to display
# FIXME - DOES NOT WORK FOR EXGAUSSIAN
  def plot_proposal(self, mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accepted, trace, i):
    trace = copy(trace)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16,4))
    fig.suptitle('iteration %i' %(i+1))
    x = np.linspace(-3, 3, 5000)
    color = 'g' if accepted else 'r'
        # plot prior P(theta)
    prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
    prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
    prior = norm(mu_prior_mu, mu_prior_sd).pdf(x)
    ax1.plot(x, prior)
    ax1.plot([mu_current]*2, [0, prior_current], marker = 'o', color=color)
    ax1.plot([mu_proposal]*2, [0, prior_proposal], marker = 'o', color=color)
    ax1.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), arrowprops=dict(arrowstyle="->", lw=2.0))
    ax1.set(ylabel='probability density', title='current: prior(mu=%.2f)=%.2f\nproposal: prior(mu=%.2f)=%.2f' %(mu_current, prior_current, mu_proposal, prior_proposal))
        # likelihood P(X | theta)
    likelihood_current = norm(mu_current, 1).pdf(data).prod()
    likelihood_proposal = norm(mu_proposal,1).pdf(data).prod()
    y = norm(loc=mu_proposal, scale=1).pdf(x)
    sns.distplot(data, kde=False, norm_hist=True, ax=ax2)
    ax2.plot(x,y,color=color)
    ax2.axvline(mu_current, color='b', linestyle='--', label='mu_current')
    ax2.axvline(mu_proposal, color=color, linestyle='--', label='mu_proposal')
    ax2.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), arrowprops=dict(arrowstyle="->", lw=2.0))
    ax2.set(title='likelihood(mu=%.2f)=%.2f\n likelihood(mu=%.2f)=%.2f' %(mu_current, 1e14*likelihood_current, mu_proposal, 1e14*likelihood_proposal))

        # Posterior distribution P(theta |X)
    posterior_analytical = calc_posterior_analytical(data, x, mu_prior_mu, mu_prior_sd)
    ax3.plot(x, posterior_analytical)
    posterior_current = calc_posterior_analytical(data, mu_current, mu_prior_mu, mu_prior_sd)
    posterior_proposal = calc_posterior_analytical(data, mu_proposal, mu_prior_mu, mu_prior_sd)
    ax3.plot([mu_current]*2, [0, posterior_current], marker='o', color='b')
    ax3.plot([mu_proposal]*2, [0, posterior_proposal], marker='o', color=color)
    ax3.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), arrowprops=dict(arrowstyle="->",lw=2.0))
    ax3.set(title='posterior(mu=%.2f)=%.5f\n posterior(mu=%.2f)=%.5f' %(mu_current, posterior_current, mu_proposal, posterior_proposal))
    if accepted:
      trace.append(mu_proposal)
    else:
      trace.append(mu_current)
    ax4.plot(trace)
    ax4.set(xlabel='iteration', ylabel='mu', title='trace')
    plt.tight_layout()
    plt.show()
