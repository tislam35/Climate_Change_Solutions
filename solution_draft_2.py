# Code based on a modified version of https://www.kaggle.com/vijaikm/co2-emission-forecast-with-python-seasonal-arima
# Trends data taken from https://www.iea.org/reports/global-ev-outlook-2021/trends-and-developments-in-electric-vehicle-markets

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

path = "Datasets/USCarSales.csv"

dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
mte = pd.read_csv(path, parse_dates=['Year'], date_parser=dateparse) 
mte['Year'] = mte['Year'].dt.year
mte.set_index('Year', inplace=True)
mte.index = pd.to_datetime(mte.index, format='%Y', errors = 'coerce')
mte.index = mte.index.to_period('Y')

mod = sm.tsa.arima.ARIMA(mte, 
                                order=(2,2,2),  
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
      
# Get forecast of 10 years or 120 months steps ahead in future
forecast = results.get_forecast(steps=10)
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()

ax = mte.plot(label='Observed Car Sales', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecasted Car Sales')

linear_end = forecast.predicted_mean[9] / 2

path = "Datasets/USEVsales.csv"

dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
mte = pd.read_csv(path, parse_dates=['Year'], date_parser=dateparse) 
mte['Year'] = mte['Year'].dt.year
mte.set_index('Year', inplace=True)
mte.index = pd.to_datetime(mte.index, format='%Y', errors = 'coerce')
mte.index = mte.index.to_period('Y')
mte_2 = mte

mod = sm.tsa.arima.ARIMA(mte, 
                                order=(3,0,0),  
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
      
# Get forecast of 10 years or 120 months steps ahead in future
forecast = results.get_forecast(steps=10)
actuals = forecast
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()

mte.plot(ax=ax, label='Observed EV sales')
forecast.predicted_mean.plot(ax=ax, label='Forecasted EV sales')

linear_start = forecast.predicted_mean[0]
gap = (linear_end - linear_start) / 9

expected = [linear_start,]
for i in range(0,9):
    expected.append(linear_start + (gap * (i+1)))   
    
expected_exp = np.logspace(np.log(linear_start), np.log(linear_end), 10, base=np.exp(1))
expected_s1 = np.logspace(np.log(linear_start), np.log((linear_end - linear_start) * 0.67), 5, base=np.exp(1))
expected_s2 = np.logspace(np.log((linear_end - linear_start) * 0.67), np.log(linear_end), 6, base=np.exp(1))
print(expected_s1)
print(expected_s2)
    
years = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
plt.plot(years, expected, label='Target EV sales (linear growth)')
plt.plot(years, expected_exp, label='Target EV sales (exp growth)')
plt.plot(years, list(expected_s1) + list(expected_s2)[1:], label='Target EV sales (s growth)')

ax.set_xlabel('Year')
ax.set_ylabel('Car Sales in Thousands')

path = "Datasets/USEmissions.csv"

dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
mte = pd.read_csv(path, parse_dates=['Year'], date_parser=dateparse) 
mte['Year'] = mte['Year'].dt.year
mte.set_index('Year', inplace=True)
mte.index = pd.to_datetime(mte.index, format='%Y', errors = 'coerce')
mte.index = mte.index.to_period('Y')

mod = sm.tsa.arima.ARIMA(mte, 
                                order=(1,1,3),  
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())

# Get forecast of 10 years or 120 months steps ahead in future
forecast = results.get_forecast(steps=10)
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
print(forecast_ci.head())

ax = mte.plot(label='observed', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Car Sales in Thousands')

actuals_list = actuals.predicted_mean
combined_s = list(expected_s1).copy() 
combined_s.extend(expected_s2[1:])
mte_2_list = mte_2.values.tolist()

for i in range(0, 10):
    actuals_list[i] -= mte_2_list[i]
    expected[i] -= mte_2_list[i]
    expected_exp[i] -= mte_2_list[i]
    combined_s[i] -= mte_2_list[i]

new_forecast = forecast.predicted_mean - ((np.cumsum(actuals_list) * 4600) / 1000000)
plt.plot(years, list(new_forecast), label='Forecast minus EVs')
plt.plot(years, list(forecast.predicted_mean) - ((np.cumsum(expected) * 4600) / 1000000), label='Forecast minus Expected')
plt.plot(years, list(forecast.predicted_mean) - ((np.cumsum(expected_exp) * 4600) / 1000000), label='Forecast minus Expected Log')
plt.plot(years, list(forecast.predicted_mean) - ((np.cumsum(combined_s) * 4600) / 1000000), label='Forecast minus Expected S')

x = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
fig1, ax1 = plt.subplots(figsize=(20, 15))
y = [sales for sublist in mte_2_list for sales in sublist]
y0 = y.copy()
y0.extend(actuals_list)
y1 = y.copy()
y1.extend(expected)
y2 = y.copy()
y2.extend(expected_exp)
y3 = y.copy()
y3.extend(combined_s)

ax1.plot(x, np.cumsum(y0), label = 'Forecast')
ax1.plot(x, np.cumsum(y1), label = 'Expected with linear growth')
ax1.plot(x, np.cumsum(y2), label = 'Expected with exponential growth')
ax1.plot(x, np.cumsum(y3), label = 'Expected with s-curve growth')
ax1.set_xlabel('Year')
ax1.set_ylabel('Registered EVs in thousands')

plt.legend()
plt.show()

#co2 2030
#forecast - 4246.669349925766
#actuals - 4232.1270064141945
#expected linear - 4071.00035651
#expected exp - 4139.9093338
#expected s - 4061.61798725

#no of cars 2030
#actuals - 4,293,382
#expected linear - 39,320,915
#expected exp - 24,340,702
#expected s - 41,360,560

#cars registered 35%
#estimated no of cars by 2030: 309,309,894
#35% of est = 108,258,462.9
#15% of 4246.669349925766 = 637.0004024888649

#actuals
#market share = 4,293,382/108,258,462.9 = 3.965862700236066255841879314176%
#theoretical decrease = 25.262561362659507722345464781211
#actual decrease = 14.5423435115715

#expected linear
#market share = 39,320,915/108,258,462.9 = 36.321331327529868336972295955257%
#theoretical decrease = 231.36702674567943804958180317929
#actual decrease = 175.668993415766

#expected exp
#market share = 24,340,702/108,258,462.9 = 22.483879179481680965193160894214%
#theoretical decrease = 143.22240086840840273153185514146
#actual decrease = 106.760016125766

#expected s
#market share = 41,360,560/108,258,462.9 = 38.205382648195719024937310466838%
#theoretical decrease = 243.36844124141768161336050153729
#actual decrease = 185.051362675766

