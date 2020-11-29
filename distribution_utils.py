import numpy as np
import data_utilities as dt
import statistics as stats
import pandas as pd
import statsmodels.api as sms
import scipy.stats as sc
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import pylab
class Dist_Utils(object):

     DATA = pd.read_csv("S&P500(1970-2020).csv")
     RETURNS = DATA["Daily_Return_Pct"].dropna()
     CLOSE = DATA["Close"].dropna()

     mean = []

     def mean_breaks(self):
         data = dt.data_iterator(self.RETURNS,20)
         shifts = []
         for i,d in enumerate(data):
             p = self.adf_test(d)
             if p < .5:
                 shifts.append([i,p])
         print(shifts)


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

     def fitter(self,data):
         """
         Method to fit data to distribution
         :param data: data set to fit
         :return:
         """

ob = Dist_Utils()
# pd.Series.plot(ob.RETURNS)

# plt.hist(ob.RETURNS.to_numpy(), bins= 50)
# plt.savefig('hist_retruns.png')
# plt.show()
# sms.qqplot(ob.RETURNS.to_numpy(),line = '45')
# pylab.savefig('qq_plot_retruns.png')
# pylab.show()

def bond_generate( length = 1000):
    bond_noise = np.random.random(length)
    bond_price = []
    for i in list(range(length)):
        bond_price.append(np.cos(i)+ bond_noise[i])

    return bond_price


def eq_generate(length = 1000):
    eq_noise = np.random.random(length)
    eq_price = []
    for i in list(range(length)):
        eq_price.append(np.sin(i) + eq_noise[i])

    return eq_price


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



def GBM_Simulator(n):
    for i in range(n):
        N = len(ob.CLOSE.to_numpy())
        T = N
        d = 1
        t = np.arange(1,int(N)+1)
        b = np.random.normal(size = N+1)
        W = b.cumsum()
        fake,m = GBM(ob.CLOSE.to_numpy()[1],np.mean(ob.RETURNS.to_numpy()),np.std(ob.RETURNS.to_numpy()),
           W,T,N)
        plt.plot(np.arange(N+1),fake)
        plt.savefig(f"GBM number {i}")
        plt.show()

GBM_Simulator(5)