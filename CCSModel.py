import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from GlobalEmissionsARIMA import data, getModel, optimParam

#Loading in the data
dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
fileData = pd.read_csv("Datasets/CCSFacilitiesAndCapacity.csv", 
                    encoding_errors="replace", 
                    parse_dates=['Operation Date'], 
                    index_col='Operation Date', 
                    date_parser=dateparse)
CCSdata = fileData.groupby(["Operation Date"]).sum()
CCSdata.info()

#Figure of current and future CCS capacities
fig, ax = plt.subplots()
bars = ax.bar(CCSdata.index, CCSdata["Max Storage"], 250)
ax.bar_label(bars, fileData.groupby(["Operation Date"]).count()["Title"])
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("CO2 Max Storage Capacity (Million Tones)")
ax.set_title("Number of CCS Facilities and Their CO2 Storage Capacity Since 1972")
plt.show()

#Cumulitive capacities
fig, ax = plt.subplots()
ax.plot(CCSdata.cumsum(), marker='o')
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("CO2 Max Storage Capacity (Million Tones)")
ax.set_title("Total Carbon Emissions Storage Capacity Since 1972")
plt.show()

#emissions data
data.info()

#combined model draft
def basicPrediction():
    results = ARIMA(data, order=optimParam(data), enforce_stationarity=False, enforce_invertibility=False).fit()
    predictions = results.get_prediction(start=data.index[0], end=str(2028), dynamic=False)
    test = pd.DataFrame(predictions.predicted_mean)
    test = test.subtract(CCSdata.cumsum(), axis=0, fill_value=0)
    test["Max Storage"] = test["Max Storage"].fillna(0)
    test["Adjusted Prediction"] = test["Max Storage"] + test["predicted_mean"]
    fig, ax = plt.subplots()
    ax.plot(test.index, test["Adjusted Prediction"], label="With CCS", color='g')
    ax.plot(test.index, test["predicted_mean"], label="Without CCS", color='r', alpha=0.5)
    ax.set_xlabel("Time (Yearly)")
    ax.set_ylabel("Million tonnes of carbon dioxide")
    ax.set_title("Global carbon emissions predictions based on observed data")
    plt.legend()
    plt.show()

#developing a basic predictive model
average = fileData["Max Storage"].mean()
print(average)
#average max capacity over last 50 years is 1.3345583333333333
def predictWithCCS(sitesPerYear, targetYear):
    addedMax = sitesPerYear * 1.3345583333333333
    if(targetYear < 2029):
        basicPrediction()
    else:
        indexes = pd.date_range(start="01/01/2029", periods=targetYear-2028, freq='YS')
        storage = CCSdata.cumsum()["Max Storage"]
        curMax = storage.iloc[-1]
        vals = list()
        for i in indexes:
            curMax += addedMax
            vals.append(curMax)
        addedData = pd.DataFrame(vals, columns=["Max Storage"], index=indexes)
        storage = pd.DataFrame(storage)
        fullData = storage.append(addedData)
        print(fullData)
        results = ARIMA(data, order=optimParam(data), enforce_stationarity=False, enforce_invertibility=False).fit()
        predictions = results.get_prediction(start=data.index[0], end=str(targetYear), dynamic=False)
        test = pd.DataFrame(predictions.predicted_mean)
        test = test.subtract(fullData, axis=0, fill_value=0)
        test["Max Storage"] = test["Max Storage"].fillna(0)
        test["Adjusted Prediction"] = test["Max Storage"] + test["predicted_mean"]
        fig, ax = plt.subplots()
        ax.plot(test.index, test["Adjusted Prediction"], label="With CCS", color='g')
        ax.plot(test.index, test["predicted_mean"], label="Without CCS", color='r', alpha=0.5)
        ax.set_xlabel("Time (Yearly)")
        ax.set_ylabel("Million tonnes of carbon dioxide")
        ax.set_title("Global carbon emissions predictions based on average max capacity")
        plt.legend()
        plt.show()

predictWithCCS(20, 2050)