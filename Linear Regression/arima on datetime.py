# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:39:54 2019

@author: P70002567
"""
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

series = pd.read_excel('C:\\Users\\P70002567\\Desktop\\Linear Regression\\Calls.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=4)[0]
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = sqrt(mean_squared_error(test, predictions))
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()




