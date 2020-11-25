import numpy as np
import data_utilities as dt
import scipy.stats as stats
import pandas as pd
from matplotlib import pyplot as plt

class Dist_Utils(object):


    def __init__(self):
        self.data = pd.read_csv("S&P500(1970-2020).csv")
        self.returns = self.data["Daily_Return_Pct"]
        self.close = self.data["Close"]

    def approximator(self):
        prices = dt.slice(self.close,1,10000)
        data = dt.slice(self.returns,1,100)
        loc,scale = stats.expon.fit(prices)
        mu,sigma = stats.norm.fit(data)
        # data = dt.data_iterator(self.returns, 20)
        # means = [pd.DataFrame.mean(x)for x in data]
        plt.hist(prices, bins=50, density=True, alpha=0.6, color='g')
        # xmin,xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = stats.norm.pdf(x, mu, sigma)
        # plt.plot(x, p, 'k', linewidth=2)
        # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
        # plt.title(title)
        plt.show()


    def confidence_interval(self):
      pass

    def fitter(self,data):
         """
         Method to fit data to distribution
         :param data: data set to fit
         :return:
         """


dis = Dist_Utils().approximator()

