from os import stat
import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import warnings
import itertools
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA

#Loading in the data
dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
data = pd.read_csv("Datasets/GlobalEmissions.csv", parse_dates=['Year'], index_col='Year', date_parser=dateparse)
#print(data.index, data.values)
#data.info()

'''
#Testing data with basic graph
fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global carbon emissions since 1965")
plt.show()

#Making data stationary
movAve = data.rolling(5).mean()
fig, ax = plt.subplots()
ax.plot(data)
ax.plot(movAve, color="red")
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global carbon emissions since 1965")
plt.show()
stationary_data = data - movAve
fig, ax = plt.subplots()
ax.plot(data)
ax.plot(movAve, color="red")
ax.plot(stationary_data, color="green")
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global carbon emissions since 1965")
plt.show()
'''

#Finding optimal p, d, q values
def optimParam(curData):
    lowest_aic = float("inf")
    optimal_params = list()
    results = None
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p,d,q))
    for i in pdq:
        model = ARIMA(curData, order=i, enforce_stationarity=False, enforce_invertibility=False)
        accuracy = model.fit().aic
        if(accuracy < lowest_aic):
            lowest_aic = accuracy
            optimal_params = i
            results = model.fit()
    results.plot_diagnostics(figsize=(15,12))
    plt.show()
    return optimal_params

'''
#Getting ARIMA model forecast for given number of years
def getModel(years):
    predictions = list()
    values = data.values
    for i in range(0,years):
        model = ARIMA(values, order=optimParam(values), enforce_stationarity=False, enforce_invertibility=False)
        nextGuess = model.fit().forecast()[0]
        predictions.append(nextGuess)
        values= np.append(values, nextGuess)
    fig, ax = plt.subplots()
    ax.plot(data)
    dates = pd.date_range(start="01/01/2020", periods=years, freq='YS')
    ax.plot(dates, predictions, color="red")
    ax.set_xlabel("Time (Yearly)")
    ax.set_ylabel("Million tonnes of carbon dioxide")
    ax.set_title("Global carbon emissions since 1965")
    plt.show()

getModel(10)
'''

def getModel(targetYear):
    results = ARIMA(data, order=optimParam(data), enforce_stationarity=False, enforce_invertibility=False).fit()
    predictions = results.get_prediction(start=data.index[0], end=str(targetYear), dynamic=False)
    fig, ax = plt.subplots()
    ax.plot(data, label="Observed")
    ax.plot(predictions.predicted_mean, label="Predicted")
    ax.fill_between(predictions.conf_int().index, predictions.conf_int().iloc[:, 0], predictions.conf_int().iloc[:, 1], color='green', alpha=.5)
    ax.set_xlabel("Time (Yearly)")
    ax.set_ylabel("Million tonnes of carbon dioxide")
    ax.set_title("Global carbon emissions since 1965")
    plt.legend()
    plt.show()

#getModel(2050)