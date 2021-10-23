import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 16
import warnings
import itertools
warnings.filterwarnings("ignore")

#Loading in the data
dateparse = lambda x: pd.to_datetime(x, format='%Y', errors = 'coerce')
data = pd.read_csv("Datasets/GlobalEmissions.csv", parse_dates=['Year'], index_col='Year', date_parser=dateparse)
print(data.index, data.values)
data.info()

#Testing data with basic graph
fig, ax = plt.subplots()
ax.plot(data.index, data.values)
ax.set_xlabel("Time (Yearly)")
ax.set_ylabel("Million tonnes of carbon dioxide")
ax.set_title("Global carbon emissions since 1965")
plt.show()

#