# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:45:13 2020

@author: P70002567
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:03:19 2020

@author: P70002567
"""



from statsmodels.tsa.arima_model import ARIMA
import numpy
import pandas as pd

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load dataset
series1 = pd.read_excel('C:\\Users\\P70002567\\Desktop\\Linear Regression\\Calls.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True)
series = series1[0:24]
series.astype('float64', raise_on_error = False)
# seasonal difference
X = series.values
months_in_year = 12
differenced = difference(X, months_in_year)
# fit model
model = ARIMA(differenced, order=(5,0,0))
model_fit = model.fit(disp=-1)
# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=12)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, months_in_year)
	print('Month %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1