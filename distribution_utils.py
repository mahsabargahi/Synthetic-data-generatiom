import warnings
import scipy.stats as st
import matplotlib
import numpy as np
import data_utilities as dt
import statistics as stats
import pandas as pd
import statsmodels.api as sms

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.seasonal import seasonal_decompose
class Dist_Utils(object):

     DATA = pd.read_csv("S&P500(1970-2020).csv")
     RETURNS = DATA["Daily_Return_Pct"].dropna()
     CLOSE = DATA["Close"]
     plt.plot(np.arange(len(CLOSE.to_numpy())), CLOSE.to_numpy())
     plt.savefig(f"SPCLOSEnumber")
     plt.show()
     NORM_MODEL = st.norm.fit(RETURNS)
     LAP_MODEL = st.laplace.fit(RETURNS)
     LOGRETURNS = np.log(RETURNS+(-1*RETURNS.min()+1e-5)).dropna()
     LOGNORM_MODEL = st.norm.fit(LOGRETURNS)
     CLOSE = DATA["Close"].dropna()

     mean = []

     def mean_breaks(self):
         data = dt.data_iterator(self.CLOSE,10)
         means = []
         sigmas = []
         breaks = []
         for i,d in enumerate(data):
             p = np.mean(d.to_numpy())
             q = np.std(d.to_numpy())
             means.append(p)
             sigmas.append(q)
         for i in range(len(means)-1):
             if 1-self.z_test(mu = means[i], x_bar = means[i+1],sigma = sigmas[i],n = 10) < .05:
                 breaks.append(i+1)
         increase = []
         for i in range(len(breaks)-1):
             increase.append(means[breaks[i+1]]-means[breaks[i]])
         return breaks,increase,len(breaks)/len(means)

     def z_test(self, mu, x_bar, sigma, n):
         z = (mu - x_bar)/(sigma/np.sqrt(n))
         return st.norm.cdf(z)

     def adf_test(self,timeseries):
         # Perform Dickey-Fuller test:
         print('Results of Dickey-Fuller Test:')
         dftest = adfuller(timeseries, autolag='AIC')
         dfoutput = pd.Series(dftest[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
         for key, value in dftest[4].items():
             dfoutput['Critical Value (%s)' % key] = value
         print(dfoutput)
         return dftest['p-value']


     def kpss_test(self,series, **kw):
         statistic, p_value, n_lags, critical_values = kpss(series, **kw)
         # Format Output
         print(f'KPSS Statistic: {statistic}')
         print(f'p-value: {p_value}')
         print(f'num lags: {n_lags}')
         print('Critial Values:')
         for key, value in critical_values.items():
             print(f'   {key} : {value}')
         print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
         return p_value



ob = Dist_Utils()
parameter = ob.NORM_MODEL


def truncated_fat_tails(mu, sigma,size):
    min = mu-2*sigma
    max = mu+2*sigma
    return_values = []
    fig = sms.qqplot(ob.LOGRETURNS,dist = st.laplace)
    plt.show()
    fig = sms.qqplot(ob.RETURNS, dist=st.cauchy)
    plt.show()
    fig = sms.qqplot(ob.LOGRETURNS, dist=st.norm)
    plt.show()
    num = ob.RETURNS[ob.RETURNS > max].count()
    kurt = st.kurtosis(ob.RETURNS)
    skew = st.skew(ob.RETURNS)
    while len(return_values) < size:
        fat = .01
        test = np.random.random()
        if test < fat:
            value = np.random.normal(loc = mu, scale = 2*sigma)
            if value > min and value < max:
                return_values.append(value)
        else:
            value = np.random.normal(loc = mu, scale = sigma)
            if value > min and value < max:
                return_values.append(value)
            else:
                sig_3 = .001
                t = np.random.random()
                if t < sig_3: return_values.append(value)
    return return_values


g = truncated_fat_tails( mu = parameter[0], sigma = parameter[1], size = 10000)
plt.hist(g, bins = 100)
plt.show()
returns = ob.RETURNS.tolist()
plt.hist(returns, bins = 100 )
plt.show()
def GBM(So, mu, sigma, W, T, N):

    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in np.arange(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i - 1]
        S_temp = So * np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

def Johnston(So, N):
    data = st.johnsonsu.rvs(a = .05, b = 1.2, loc = .0001, scale = .01, size = N)
    S = []
    S.append(So)
    for i in np.arange(0, int(N)):
        S.append(S[i]+data[i])
    return S

def GBM_With_Trend(So, mu, sigma, W, T, N):
    breaks, diff,lamda = ob.mean_breaks()
    pois = np.random.poisson(lamda/100,N)
    interest_rate = np.random.uniform(.01, .03, size=50)
    t = np.linspace(0., 1., N + 1)
    S = []
    S.append(So)
    trend = interest_rate[random.randint(0,len(interest_rate)-1)]
    for i in np.arange(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i - 1]
        S_temp = So * np.exp(drift + diffusion)
        if pois[i-1] == 1:
           trend = trend = interest_rate[random.randint(0,len(interest_rate)-1)]
           S.append(S_temp+trend)
        else: S.append(S_temp+trend)
    return S, t

def GBM_Simulator(n):
    for i in range(n):
        N = 12000
        T = N
        d = 1
        t = np.arange(1,int(N)+1)
        b = np.random.normal(loc = 0, scale= .01, size = N)
        W = b.cumsum()
        fake_gbm,m = GBM_With_Trend(ob.CLOSE.to_numpy()[1],np.mean(ob.RETURNS.to_numpy()),np.std(ob.RETURNS.to_numpy()),
           W,T,N)
        fake_jon = Johnston(ob.CLOSE.to_numpy()[1],N)
        plt.hist(ob.RETURNS, bins = 200)
        plt.savefig("Real Hist")
        plt.show()
        fake_jon = pd.Series(fake_jon)
        fake_gbm = pd.Series(fake_gbm)
        fake_returns = fake_gbm.pct_change(periods = 1).dropna()
        fake_jonRetuns = fake_jon.pct_change(periods=1).dropna()
        plt.hist(fake_returns, bins = 200)
        plt.savefig("Sythentic GBM Data Hist")
        plt.show()
        plt.hist(fake_jonRetuns, bins=200)
        plt.savefig("Sythentic John Data Hist")
        plt.show()
        plt.plot(np.arange(N+1),fake_gbm)
        plt.savefig(f"GBM number {i}")
        plt.show()
        plt.plot(np.arange(N + 1), fake_jon)
        plt.savefig(f"Jon number {i}")
        plt.show()



GBM_Simulator(5)

# def estimate_variance(k=5):
#
#     prices = ob.CLOSE.to_numpy()
#     log_prices = np.log(prices)
#     rets = np.diff(log_prices)
#     T = len(rets)
#     mu = np.mean(rets)
#     var_1 = np.var(rets, ddof=1, dtype=np.float64)
#     rets_k = (log_prices - np.roll(log_prices, k))[k:]
#     m = k * (T - k + 1) * (1 - k / T)
#     var_k = 1/m * np.sum(np.square(rets_k - k * mu))
#
#     # Variance Ratio
#     vr = var_k / var_1
#     # Phi1
#     phi1 = 2 * (2*k - 1) * (k-1) / (3*k*T)
#     # Phi2
#
#     def delta(j):
#         res = 0
#         for t in range(j+1, T+1):
#             t -= 1  # array index is t-1 for t-th element
#             res += np.square((rets[t]-mu)*(rets[t-j]-mu))
#         return res / ((T-1) * var_1)**2
#
#     phi2 = 0
#     for j in range(1, k):
#         phi2 += (2*(k-j)/k)**2 * delta(j)
#
#     return vr, (vr - 1) / np.sqrt(phi1), (vr - 1) / np.sqrt(phi2)
# ran = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 200, 500, 1000]
#
# for i in ran:
#     vr,z,z_2 = estimate_variance(i)
#     print (vr,sc.norm.cdf(z),i)



 # params = st.johnsonsu.fit(fake_returns)
        # arg = params[:-2]
        # loc = params[-2]
        # scale = params[-1]
        # y, x = np.histogram(fake_returns, bins=200, density=True)
        # # Calculate fitted PDF and error with fit in distribution
        # pdf = st.johnsonsu.pdf(x, loc=loc, scale=scale, *arg)
        # plt.figure(figsize=(12, 8))
        # ax = ob.RETURNS.plot(kind='hist', bins=50, normed=True, alpha=0.5)
        # # Save plot limits
        # dataYLim = ax.get_ylim()
        # ax.set_ylim(dataYLim)
        # ax.set_title(u'Retruns.\n All Fitted Distributions')
        # ax.set_xlabel(u'date')
        # ax.set_ylabel('retrun')
        # pdf = make_pdf(st.johnsonsu, params)
        # # Display
        # plt.figure(figsize=(12, 8))
        # ax = pdf.plot(lw=2, label='PDF', legend=True)
        # ob.RETURNS.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
        # ax.set_xlabel(u'Temp. (°C)')
        # ax.set_ylabel('Frequency')
        # plt.savefig("optimal pdf for data")
        # plt.show()
        #

# matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
# matplotlib.style.use('ggplot')
#
# # Create models from data
# def best_fit_distribution(data, bins=200, ax=None):
#     """Model data by finding best fit distribution to data"""
#     # Get histogram of original data
#     y, x = np.histogram(data, bins=bins, density=True)
#     x = (x + np.roll(x, -1))[:-1] / 2.0
#
#     # Distributions to check
#     DISTRIBUTIONS = [
#         st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
#         st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
#         st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
#         st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
#         st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
#         st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,
#         st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,
#         st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,st.t,st.triang,st.truncexpon,st.truncnorm,
#         st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
#     ]
#
#     # Best holders
#     best_distribution = st.norm
#     best_params = (0.0, 1.0)
#     best_sse = np.inf
#
#     # Estimate distribution parameters from data
#     for distribution in DISTRIBUTIONS:
#         print(distribution.name)
#         # Try to fit the distribution
#         try:
#             # Ignore warnings from data that can't be fit
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore')
#
#                 # fit dist to data
#                 params = distribution.fit(data)
#
#                 # Separate parts of parameters
#                 arg = params[:-2]
#                 loc = params[-2]
#                 scale = params[-1]
#
#                 # Calculate fitted PDF and error with fit in distribution
#                 pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
#                 sse = np.sum(np.power(y - pdf, 2.0))
#
#                 # if axis pass in add to plot
#                 try:
#                     if ax:
#                         pd.Series(pdf, x).plot(ax=ax)
#
#                 except Exception:
#                     pass
#
#                 # identify if this distribution is better
#                 if best_sse > sse > 0:
#                     best_distribution = distribution
#                     print(best_distribution.name)
#                     best_params = params
#                     best_sse = sse
#
#         except Exception:
#             pass
#
#     return (best_distribution.name, best_params)
#
# def make_pdf(dist, params, size=10000):
#     """Generate distributions's Probability Distribution Function """
#
#     # Separate parts of parameters
#     arg = params[:-2]
#     loc = params[-2]
#     scale = params[-1]
#
#     # Get sane start and end points of distribution
#     start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
#     end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
#
#     # Build PDF and turn into pandas Series
#     x = np.linspace(start, end, size)
#     y = dist.pdf(x, loc=loc, scale=scale, *arg)
#     pdf = pd.Series(y, x)
#
#     return pdf
# plt.figure(figsize=(12,8))
# ax = ob.RETURNS.plot(kind='hist', bins=50, normed=True, alpha=0.5)
# # Save plot limits
# dataYLim = ax.get_ylim()
# best_fit_name, best_fit_params = best_fit_distribution(ob.RETURNS, 200, ax)
# best_dist = getattr(st, best_fit_name)
# ax.set_ylim(dataYLim)
# ax.set_title(u'Retruns.\n All Fitted Distributions')
# ax.set_xlabel(u'date')
# ax.set_ylabel('retrun')
# pdf = make_pdf(best_dist, best_fit_params)
#
# # Display
# plt.figure(figsize=(12,8))
# ax = pdf.plot(lw=2, label='PDF', legend=True)
# ob.RETURNS.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
#
# param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
# param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
# dist_str = '{}({})'.format(best_fit_name, param_str)
#
# ax.set_title(u'Returns \n' + dist_str)
# ax.set_xlabel(u'Temp. (°C)')
# ax.set_ylabel('Frequency')
# plt.savefig("optimal pdf for data")
# plt.show()