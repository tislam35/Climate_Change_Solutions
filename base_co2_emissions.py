from IPython import get_ipython


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

from statsmodels.tsa.arima_model import ARMA
import itertools


url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df = pd.read_csv(url, delimiter=',')



df_average_co2 = df[['iso_code', 'country', 'co2']]
df_average_co2 = df_average_co2.groupby(["country"], as_index=False).mean()



# Average annual production-based emissions of carbon dioxide (CO2)
fig = px.choropleth(df_average_co2, locations="country", locationmode="country names", color="co2", 
hover_name='country',color_continuous_scale=px.colors.sequential.Turbo, scope="world")
  
fig.show()


df_countries = df[['year', 'country', 'co2']]


len(df_countries[df_countries.co2.isnull()])



df_united_states = df_countries[df_countries.country == 'United States']


df_united_states = df_united_states.drop('country',axis=1)
df_united_states.index = pd.to_datetime(df_united_states.year, format= '%Y')
df_united_states


df_united_states = df_united_states.drop('year',axis=1)



df_united_states['Ticks'] = range(0,len(df_united_states.index.values))

# Time Series Plot
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Ticks')
ax1.set_ylabel('CO2 Emissions.')
ax1.set_title('Original Plot')
ax1.plot('Ticks', 'co2', data = df_united_states);


# Stationary test
from statsmodels.tsa.stattools import adfuller
def stationarity_check(ts):
    
    # Determing rolling statistics
    roll_mean = ts.rolling(center=False,window=12).mean()
    # Plot rolling statistics:
    plt.plot(ts, color='green',label='Original')
    plt.plot(roll_mean, color='blue', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show(block=False)
    
    # Perform Augmented Dickey-Fuller test:
    print('Augmented Dickey-Fuller test:')
    df_test = adfuller(ts)
    print("type of df_test: ",type(df_test))
    print("df_test: ",df_test)
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print("df_output: \n",df_output)
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print(df_output)


stationarity_check(df_united_states.co2)


# Comparing the test statistic to the critical values, it looks like we would have to fail to reject 
# the null hypothesis that the time series is non-stationary and does have time-dependent structure.

# Transforming non-stationary data using differencing
df_united_states["diff_1"] = df_united_states["co2"].diff(periods=1)
df_united_states["diff_2"] = df_united_states["co2"].diff(periods=2)
df_united_states["diff_3"] = df_united_states["co2"].diff(periods=3)

df_united_states.head(6)



# After differencing, the p-value is < 0.05. Now safe to reject null hypothesis and conclude
# that the data is stationary
df_test = adfuller(df_united_states["diff_1"].dropna())
df_test[1]


# Autocorrelation plots
plot_acf(df_united_states["diff_1"].dropna(), lags=50)
plot_pacf(df_united_states["diff_1"].dropna(), lags=50)
plt.xlabel('lags')
plt.show()



# Modeling
p = q = range(0, 4)
pq = itertools.product(p, q)
for param in pq:
    try:
        mod = ARMA(df_united_states["diff_1"].dropna(),order=param)
        results = mod.fit()
        print('ARMA{} - AIC:{}'.format(param, results.aic))
    except:
        continue


# Mean Square Error plot
model = ARMA(df_united_states["diff_1"].dropna(), order=(2,3))  
results_MA = model.fit()  
plt.plot(df_united_states["diff_1"].dropna())
plt.plot(results_MA.fittedvalues, color='red')
plt.title('Fitting data _ MSE: %.2f'% (((results_MA.fittedvalues-df_united_states["diff_1"].dropna())**2).mean()))
plt.show()



predictions = results_MA.predict(end = "01/01/2022" )
predictions


