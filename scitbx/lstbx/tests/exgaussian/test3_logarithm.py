import numpy as np
import matplotlib.pyplot as plt
import math
# http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
# https://github.com/mwaskom/seaborn/issues/351 for seaborn issues
# Now try varying sigma and mu !!
# Implemented using logarithm for simplicity

class mcmc():
  def __init__(self):
    pass

  def gauss_pdf(self,x,mu,sigma):
    return np.log(1./(np.sqrt(2.0*np.pi*sigma*sigma))) - (((x-mu)*(x-mu))/(2.0*sigma*sigma))

  def exgauss(self,data,mu, sigma, tau):
    sumofprob = 0.0
    #  mu +=1e-15
    #  sigma +=1e-15
    #  tau +=1e-15
#    import warnings
#    warnings.simplefilter("error", RuntimeWarning)
    ct = 0
    for x in data:
      if (1.0-math.erf((sigma*sigma-tau*(x-mu)))) > 0.0 :
        tmp_term= np.log(1./(2*tau))+((sigma*sigma-2.0*tau*(x-mu))/(2.0*tau*tau)) + np.log(1.0-math.erf((sigma*sigma-tau*(x-mu))/(sigma*tau*math.sqrt(2))))
#        if np.isnan(tmp_term):
#          sumofprob += -1000000000.0
#          ct += 1
#          continue
#          from IPython import embed; embed(); exit()
        sumofprob += tmp_term
#    print 'what is copunt ??',ct
    return sumofprob

# FIXME
  def initial_guess0(self, data, wiki_method=False):
    gamma = (np.mean(data)-np.median(data))/np.std(data)
    if (wiki_method):
#      from IPython import embed; embed(); exit()
      m = np.mean(data)
      s = np.std(data)
      mu = m-s*np.sign(gamma)*((np.abs(gamma)/2.)**(1./3))
      sigma = np.sqrt(s*s*(1-((np.abs(gamma)/2.)**(2./3))))
      tau = np.sign(gamma)*s*((np.abs(gamma)/2.)**1./3)
    else:
      mu = np.mean(data) - gamma #skewness(data)
      tau = np.std(data)*0.8
      sigma = np.sqrt(np.var(data)-tau*tau)
    return [mu,sigma,tau]

  def sampler(self, data, samples=5, mu_init=0.5, sigma_init = 0.5,tau_init = 0.5, 
              proposal_width = 10.0,t_start=1, dt=1,cdf_cutoff=0.95, 
              plot=False,seed=22):
    np.random.seed(seed) # was with seedhere
    mu_current = mu_init
    sigma_current = sigma_init
    tau_current = tau_init
    mu_alt, sigma_alt, tau_alt = self.initial_guess0(data, wiki_method=False)
    # FIXME this is terrible but if the minimization fails, MCMC should still go on !!!
    if (mu_alt == mu_init) and (sigma_alt == sigma_init) and (tau_alt == tau_init):
      mu_alt, sigma_alt, tau_alt = self.initial_guess0(data, wiki_method=True)
#
    mu_prior_mu = mu_init
    mu_prior_sd = (np.abs(mu_alt-mu_init))/2
    sd_prior_mu = sigma_init
    sd_prior_sd = (np.abs(sigma_alt-sigma_init))/2
    tau_prior_mu = tau_init
    tau_prior_sd = np.abs(tau_alt-tau_init)/2
    accept_counter = 0
#
    posterior = [[mu_current,sigma_current, tau_current]]
    data = np.sort(data)
    maxI = np.max(data) #+np.max(data)/2.
    minI = np.min(data) #-np.min(data)/2.
    I_mcmc = []
    from construct_random_datapt import ExGauss
#    print 'Starting likelihood, sd, and # of data pts = ', self.exgauss(data, mu_current, sigma_current, tau_current), mu_prior_sd, sd_prior_sd, tau_prior_sd, len(data)
    for i in range(samples):
#      print 'numstep = ',i
# trial move
      mu_proposal =  np.random.normal(mu_current,1.3* mu_prior_sd/1.)
      sigma_proposal = np.abs(np.random.normal(sigma_current, 1.3*sd_prior_sd/1.)) #
      tau_proposal = np.abs(np.random.normal(tau_current, 1.3*tau_prior_sd/1.))
      likelihood_current = self.exgauss(data,mu_current, sigma_current, tau_current) # multiply the probabilties
      likelihood_proposal = self.exgauss(data,mu_proposal,sigma_proposal, tau_proposal) # multiply

      prior_current = self.gauss_pdf(mu_current, mu_prior_mu, mu_prior_sd)+self.gauss_pdf(sigma_current, sd_prior_mu, sd_prior_sd)+ \
                      self.gauss_pdf(tau_current, tau_prior_mu, tau_prior_sd)
      prior_proposal = self.gauss_pdf(mu_proposal, mu_prior_mu, mu_prior_sd)+self.gauss_pdf(sigma_proposal, sd_prior_mu, sd_prior_sd)+ \
                       self.gauss_pdf(tau_proposal, tau_prior_mu, tau_prior_sd)
#               prior_proposal = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)*norm(sd_prior_mu, sd_prior_sd).pdf(sigma_proposal)*norm(tau_prior_mu, tau_prior_sd).pdf(tau_proposal)) #
      p_current = likelihood_current+prior_current
      p_proposal = likelihood_proposal+prior_proposal
      p_accept = p_proposal- p_current
      accept = np.log(np.random.rand()) < p_accept
      if plot:
        plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)
      if accept:
        mu_current = mu_proposal
        sigma_current = sigma_proposal
        tau_current = tau_proposal
        accept_counter +=1
      if i > t_start and i%dt:
        EXG= ExGauss(10000, minI, maxI, mu_current,sigma_current,tau_current)
        I_mcmc.append(EXG.find_x_from_iter(cdf_cutoff))
        del(EXG)
#        print 'Done collecting stats',i
#      I_mcmc.append(np.random.normal(1000., 10.))
        posterior.append([mu_current, sigma_current, tau_current])
#    print 'final likelihood', self.exgauss(data, posterior[-1][0], posterior[-1][1], posterior[-1][2])
#    print 'accept rate = %4.3f, mu_prior_sd = %12.3f, sd_prior_sd= %12.3f, tau_prior_sd= %12.3f len(d) = %6d'%(accept_counter*1.0/samples, mu_prior_sd, sd_prior_sd, tau_prior_sd, len(data)) 
#    return np.random.normal(1000., 10.), np.random.normal(100., 5.)
    print 'acceptance rate',accept_counter*1.0/samples
    return np.mean(I_mcmc), np.var(I_mcmc), accept_counter*1.0/samples
#    return posterior

# function to display
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
