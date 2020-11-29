import pandas as pd
import numpy as np
from matplotlib import pyplot as pt
import statistics as stats



#slice_slices the data by row,this function returns rows x to y
def slice(data,x,y):
    return data.iloc[int(x):int(y)-1]



def data_iterator(data,epoch = 10):
    """
    method to interate through data set
    :param epoch: number of rows per episode
    :return:
    """
    num_rows = len(data.index)
    num_iters = int(num_rows/epoch)
    data_set = []
    for i in range(num_iters):
        data_set.append(slice(data = data,x = epoch*i, y = epoch*i + epoch-1))
    return data_set


####return data by date
#data.loc[data.Date=="3/5/1970",:]

##slices by date
def dated(data,x,y):
    i = data[data['Date']==x].index[0]
    l = data[data['Date']==y].index[0]
    return slice(data,i+1,l+2)
#dated(data,"3/5/1970","3/10/1970")

# datas = slice(data,1,300)
# datas["Daily_Return_Pct"].plot()
# pt.show()
