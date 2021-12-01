# Code based on a modified version of https://www.kaggle.com/vijaikm/co2-emission-forecast-with-python-seasonal-arima
# Trends data taken from https://www.statista.com/statistics/183505/number-of-vehicles-in-the-united-states-since-1990/

import csv
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pylab
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from matplotlib.pylab import rcParams
from matplotlib import pyplot as plt 

warnings.filterwarnings("ignore") # specify to ignore warning messages

path = "Datasets/USRegistered.csv"

dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
mte = pd.read_csv(path, parse_dates=['Year'], date_parser=dateparse) 
mte['Year'] = mte['Year'].dt.year
mte.set_index('Year', inplace=True)
mte.index = pd.to_datetime(mte.index, format='%Y', errors = 'coerce')
mte.index = mte.index.to_period('Y')

print(mte)
mte.info()
'''
p = range(0, 2)
d = q = range(0, 3) # Define the p, d and q parameters to take any value between 0 and 2
pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets
pdq_x_QDQs = [(x[0], x[1], x[2], 1) for x in list(itertools.product(p, d, q))] # Generate all different combinations of seasonal p, q and q triplets
print('Examples of Seasonal ARIMA parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], pdq_x_QDQs[1]))
print('SARIMAX: {} x {}'.format(pdq[2], pdq_x_QDQs[2]))
print(pdq)
print(pdq_x_QDQs)

for param in pdq:
    for seasonal_param in pdq_x_QDQs:
        try:
            mod = sm.tsa.arima.ARIMA(mte,
                                            order=param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

a=[]
b=[]
c=[]
wf=pd.DataFrame()

for param in pdq:
    for param_seasonal in pdq_x_QDQs:
        try:
            mod = sm.tsa.arima.ARIMA(mte,
                                            order=param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            a.append(param)
            b.append(param_seasonal)
            c.append(results.aic)
        except:
            continue
wf['pdq']=a
wf['pdq_x_QDQs']=b
wf['aic']=c
print(wf[wf['aic']==wf['aic'].min()])
'''
mod = sm.tsa.arima.ARIMA(mte, 
                                order=(0,2,2),  
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())

results.plot_diagnostics(lags = 5, figsize=(20, 15))

pred = results.get_prediction(start = 6, end = 10, dynamic=False)
pred_ci = pred.conf_int()
print(pred_ci.head())

ax = mte['2012':].plot(label='observed', figsize=(20, 15))
pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='r', alpha=.5)

ax.set_xlabel('Year')
ax.set_ylabel('Registered Cars in Millions')
plt.legend()

mte_forecast = pred.predicted_mean
mte_truth = mte['2016':]

# Compute the mean square error
mse = ((mte_forecast - mte_truth['Cars']) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((mte_forecast-mte_truth['Cars'])**2)/len(mte_forecast))))
      
pred_dynamic = results.get_prediction(start=pd.to_datetime('2015'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = mte['2012':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], 
                color='r', 
                alpha=.3)

ax.fill_betweenx(ax.get_ylim(), 
                 pd.to_datetime('2015'), 
                 mte.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Year')
ax.set_ylabel('Registered Cars in Millions')

plt.legend()

# Extract the predicted and true values of our time series
mte_forecast = pred_dynamic.predicted_mean
mte_orginal = mte['2016':]

# Compute the mean square error
mse = ((mte_forecast - mte_orginal['Cars']) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((mte_forecast-mte_orginal['Cars'])**2)/len(mte_forecast))))
      
# Get forecast of 10 years or 120 months steps ahead in future
forecast = results.get_forecast(steps=11)
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
print(forecast_ci.head())

ax = mte.plot(label='observed', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='g', alpha=.4)
ax.set_xlabel('Year')
ax.set_ylabel('Registered EVs in Thousands')

print(forecast.predicted_mean[10])

plt.legend()
plt.show()