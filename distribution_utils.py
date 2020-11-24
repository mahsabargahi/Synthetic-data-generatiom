import numpy as np
import data_utilities as dt
import statistics as stats
import pandas as pd

class Dist_Utils(object):

     DATA = pd.read_csv("S&P500(1970-2020).csv")
     RETURNS = DATA["Daily_Return_Pct"]
     CLOSE = DATA["Close"]

     mean = []

     def confidence_interval(self):
      pass

     def fitter(self,data):
         """
         Method to fit data to distribution
         :param data: data set to fit
         :return:
         """


