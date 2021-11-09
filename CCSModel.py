import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
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

#predictWithCCS(20, 2050)


########################################################################################################################################


###Revisions###


#getting percentage of industry/electric sectors, setting targetyear, and getting observed CCS capacities
percents = pd.read_csv("Datasets/Global-GHG-Emissions-by-sector-based-on-WRI-2020-1.csv")["Share of global greenhouse gas emissions (%)"]
targetYear=2028
percent_by_indust_elect = (percents[7:14].sum() + percents[15:20].sum())/100
observed_max_capacities = CCSdata.cumsum()
datesCCS = pd.date_range(start=observed_max_capacities.index[0], end=observed_max_capacities.index[-1], freq='YS')
observed_max_capacities = observed_max_capacities.reindex(datesCCS, fill_value=0)
observed_max_capacities = observed_max_capacities["Max Storage"].replace(to_replace=0, method="ffill")
print(observed_max_capacities)

#Change order to pdq that provides a series with less autocorrelation
results = ARIMA(data, order=optimParam(data), enforce_stationarity=False, enforce_invertibility=False).fit()
predictions = results.get_prediction(start="2020/01/01", end=str(targetYear), dynamic=False)

#plotting global emissions for the industry and energy sectors
fig, ax = plt.subplots()
ax.plot(data, label="Observed global emissions", color='g')
ax.plot(data*percent_by_indust_elect, label="Approximate industry and energy processing emissions", color='b')
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global carbon emissions vs emissions from industry and energy sectors")
plt.legend()
plt.show()

#plotting the predicted emissions for both
fig, ax = plt.subplots()
ax.plot(predictions.predicted_mean, label="Predicted global emissions", color='g')
ax.plot(predictions.predicted_mean*percent_by_indust_elect, label="Predicted industry and energy processing emissions", color='b')
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global, industry, and energy carbon emissions predictions")
plt.legend()
plt.show()

#plotting the predicted industry/energy emissions vs the observed co2 capacity of built and upcoming CCS facilities
indexOf2020 = observed_max_capacities.index.get_loc("2020/01/01")
print(observed_max_capacities.iloc[indexOf2020:])
fig, ax = plt.subplots()
ax.plot(predictions.predicted_mean*percent_by_indust_elect, label="Predicted emissions from industry and energy processing facilities", color='b')
ax.fill_between(predictions.conf_int().index, predictions.conf_int().iloc[:, 0]*percent_by_indust_elect, predictions.conf_int().iloc[:, 1]*percent_by_indust_elect, color='green', alpha=.5)
ax.plot(observed_max_capacities.iloc[indexOf2020:], label="Observed carbon storage max capacity", color='r')
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Predicted industry and energy processing emissions vs observed CCS max capacity")
plt.legend()
plt.show()

#checking if observed CCS max capacity data is stationary by dickey-fuller test
stat_test = adfuller(observed_max_capacities.values)
print('Result: ', stat_test[0])
print('p-value: ', stat_test[1])
print('Critical Values:')
for key, value in stat_test[4].items():
	print('\t%s: %.3f' % (key, value))
#test failed : data is not stationary

#plotting prediction of CCS capacity
targetYear=2100
resultsCCS = ARIMA(observed_max_capacities, order=optimParam(observed_max_capacities), enforce_stationarity=False, enforce_invertibility=False).fit()
predictionsCCS = resultsCCS.get_prediction(start="2028/01/01", end=str(targetYear), dynamic=False)
predictions = results.get_prediction(start="2028/01/01", end=str(targetYear), dynamic=False)
fig, ax = plt.subplots()
ax.plot(predictionsCCS.predicted_mean, label="Predicted CO2 storage capacity", color='r')
ax.plot(predictions.predicted_mean, label="Predicted CO2 emissions from industry and energy processing sectors", color='b')
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Finding Net Zero")
plt.axvline(pd.to_datetime('20900617', format='%Y%m%d', errors='ignore'), color='k', linestyle='--')
plt.legend()
plt.show()