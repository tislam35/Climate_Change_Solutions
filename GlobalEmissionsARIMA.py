from os import stat
import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 16
import warnings
import itertools
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA

#Loading in the data
dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
data = pd.read_csv("Datasets/GlobalEmissions.csv", parse_dates=['Year'], index_col='Year', date_parser=dateparse)
print(data.index, data.values)
data.info()

#Testing data with basic graph
fig1, ax1 = plt.subplots()
ax1.plot(data)
ax1.set_xlabel("Time (Yearly)")
ax1.set_ylabel("Million tonnes of carbon dioxide")
ax1.set_title("Global carbon emissions since 1965")
plt.show()

#Making data stationary
movAve = data.rolling(5).mean()
fig2, ax2 = plt.subplots()
ax2.plot(data)
ax2.plot(movAve, color="red")
ax2.set_xlabel("Time (Yearly)")
ax2.set_ylabel("Million tonnes of carbon dioxide")
ax2.set_title("Global carbon emissions since 1965")
plt.show()
stationary_data = data - movAve
fig3, ax3 = plt.subplots()
ax3.plot(data)
ax3.plot(movAve, color="red")
ax3.plot(stationary_data, color="green")
ax3.set_xlabel("Time (Yearly)")
ax3.set_ylabel("Million tonnes of carbon dioxide")
ax3.set_title("Global carbon emissions since 1965")
plt.show()

#Getting ARIMA model at an arbitrary p,d,q value and forcasting next 10 years
predictions = list()
values = data.values
for i in range(0,10):
    model = ARIMA(values, order=(5,1,5))
    accuracy = model.fit()
    predictions.append(accuracy.forecast()[0])
    values= np.append(values, (accuracy.forecast()[0]))
    print(predictions)
fig4, ax4 = plt.subplots()
ax4.plot(data)
dates = pd.DatetimeIndex(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01', '2027-01-01', '2028-01-01', '2029-01-01'])
ax4.plot(dates, predictions, color="red")
ax4.set_xlabel("Time (Yearly)")
ax4.set_ylabel("Million tonnes of carbon dioxide")
ax4.set_title("Global carbon emissions since 1965")
plt.show()

#Finding optimal p, d, q values