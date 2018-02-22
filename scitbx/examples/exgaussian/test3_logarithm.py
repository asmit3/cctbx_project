import numpy as np
import matplotlib.pyplot as plt
import math
# http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
# https://github.com/mwaskom/seaborn/issues/351 for seaborn issues
# Now try varying sigma and mu !!
# Implemented using logarithm for simplicity
def calc_posterior_analytical(data, x, mu_0, sigma_0):
  sigma = 1
  n = len(data)
  mu_post = (mu_0/sigma_0**2 + data.sum() / sigma**2)/(1./sigma_0**2 + n/sigma**2)
  sigma_post = (1./sigma_0**2 + n/sigma**2)**-1
  return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

def gauss(data,mu, sigma):
  sumofprob = 0.0
  for x in data:
    sumofprob += np.log(1./(np.sqrt(2.0*np.pi*sigma*sigma))) - (((x-mu)*(x-mu))/(2.0*sigma*sigma))
  return sumofprob


def exgauss(data,mu, sigma, tau):
  sumofprob = 0.0
#  mu +=1e-15
#  sigma +=1e-15
#  tau +=1e-15
  for x in data:
    sumofprob += np.log(1./(2*tau))+((sigma*sigma-2.0*tau*(x-mu))/(2.0*tau*tau)) + np.log(1.0-math.erf((sigma*sigma-tau*(x-mu))/(sigma*tau*math.sqrt(2))))
  return sumofprob

def gauss_pdf(x,mu,sigma):
  return np.log(1./(np.sqrt(2.0*np.pi*sigma*sigma))) - (((x-mu)*(x-mu))/(2.0*sigma*sigma))

def gauss_cdf(x,mu,sigma):
  return 0.5*(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))

def exgauss_cdf(data,mu,sigma,tau):
  cdf = []
  for x in data:
    u = (x-mu)/tau
    v = sigma/tau
    cdf.append(gauss_cdf(u,0,v)-math.exp(-u+0.5*v*v+math.log(gauss_cdf(u,v*v,v))))
  return np.array(cdf)

# http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
def skewness(data):
  y_bar = np.mean(data)
  s = np.std(data)
  N = len(data)
  g = np.sum((data-y_bar)*(data-y_bar)*(data-y_bar))
  g = g/(N*s*s*s)
  return g


def sampler(data, samples=5, mu_init=0.5, sigma_init = 0.5,tau_init = 0.5, proposal_width = 10.0, plot=False,seed=22):
  print 'inside sampler', type(data)
  np.random.seed(seed)
  mu_current = mu_init
  sigma_current = sigma_init
  tau_current = tau_init
#
  mu_prior_mu = mu_init
  mu_prior_sd = proposal_width #6700.0*2.3
  sd_prior_mu = sigma_init
  sd_prior_sd = proposal_width #9500*2.3
  tau_prior_mu = tau_init
  tau_prior_sd = proposal_width #4500*2.3
  accept_counter = 0
#
  posterior = [[mu_current,sigma_current, tau_current]]
  print 'Starting likelihood', exgauss(data, mu_current, sigma_current, tau_current)
  for i in range(samples):
# trial move
    mu_proposal =  np.random.normal(mu_current, proposal_width)
    sigma_proposal = np.random.normal(sigma_current, proposal_width) #
    tau_proposal = np.random.normal(tau_current, proposal_width)
#                mu_proposal = norm(mu_current, proposal_width).rvs()       #
#               sigma_proposal = norm(sigma_current, proposal_width).rvs() #
#               tau_proposal = norm(tau_current, proposal_width).rvs()     #
                # P(X|mu,sigma) - could be gaussian, exgaussian etc
    likelihood_current = exgauss(data,mu_current, sigma_current, tau_current) # multiply the probabilties
    likelihood_proposal = exgauss(data,mu_proposal,sigma_proposal, tau_proposal) # multiply
#               print 'likelihood of proposal ',likelihood_proposal
                # P(mu), P(sigma) - we assume to be gaussian

    prior_current = gauss_pdf(mu_current, mu_prior_mu, mu_prior_sd)+gauss_pdf(sigma_current, sd_prior_mu, sd_prior_sd)+gauss_pdf(tau_current, tau_prior_mu, tau_prior_sd)
#               prior_current = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)*norm(sd_prior_mu, sd_prior_sd).pdf(sigma_current)*norm(tau_prior_mu, tau_prior_sd).pdf(tau_current)) #
    prior_proposal = gauss_pdf(mu_proposal, mu_prior_mu, mu_prior_sd)+gauss_pdf(sigma_proposal, sd_prior_mu, sd_prior_sd)+gauss_pdf(tau_proposal, tau_prior_mu, tau_prior_sd)
#               prior_proposal = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)*norm(sd_prior_mu, sd_prior_sd).pdf(sigma_proposal)*norm(tau_prior_mu, tau_prior_sd).pdf(tau_proposal)) #
    p_current = likelihood_current+prior_current
    p_proposal = likelihood_proposal+prior_proposal
    p_accept = p_proposal- p_current
#    print 'Stats ',i,likelihood_current, likelihood_proposal, prior_current, prior_proposal
#               p_accept = np.exp(p_accept)
    accept = np.log(np.random.rand()) < p_accept
    if plot:
      plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)
    if accept:
      mu_current = mu_proposal
      sigma_current = sigma_proposal
      tau_current = tau_proposal
      accept_counter +=1
    posterior.append([mu_current, sigma_current, tau_current])
  print 'final likelihood', exgauss(data, posterior[-1][0], posterior[-1][1], posterior[-1][2])
  print 'acceptance percentage = %12.7f'%(accept_counter*1.0/samples)
  return posterior

# function to display
def plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accepted, trace, i):
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
#  plt.show()
#######
plt.show()
