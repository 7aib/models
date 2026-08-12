import pandas as pd

# opsd_daily = pd.read_csv('opsd_germany_daily.csv')

# # Print the first 3 rows of the dataset
# print("First 3 rows of the dataset:")
# print(opsd_daily.head(3))

# # Print the last 3 rows of the dataset
# print("Last 3 rows of the dataset:")
# print(opsd_daily.tail(3))

# # Print the shape of the dataset
# print("Shape of the dataset:")
# print(opsd_daily.shape)

# # Print the column names of the dataset
# print("Column names of the dataset:")
# print(opsd_daily.columns)

# # Print the data types of each column
# print("Data types of each column:")
# print(opsd_daily.dtypes)    

# Set the 'Date' column as the index of the DataFrame
# opsd_daily = opsd_daily.set_index('Date') this can b done in the read_csv function itself by using the index_col parameter
opsd_daily = pd.read_csv('opsd_germany_daily.csv', index_col='Date', parse_dates=True)

opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
# opsd_daily['Weekday Name'] = opsd_daily.index.weekday_name
# Display a random sampling of 5 rows
# print(opsd_daily.sample(5, random_state=0))

# print(opsd_daily.loc['2017-08-10'])

# print(opsd_daily.loc['2014-01-20':'2014-01-22'])

# Visualizing time series data

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Display figures inline in Jupyter notebook
# %matplotlib inline

import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
opsd_daily['Consumption'].plot(linewidth=0.5);

cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
# plt.show()
plt.savefig("plot.png", dpi=300)
