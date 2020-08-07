#!/usr/local/bin/python3.7
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

# set1 ranges from jan 2015 to jul 2019, for swap rates denominated in a variety of currencies for 15 different maturities
datapath = r'/Users/desmond/OneDrive - Imperial College London/FYP/Python/cleanedscripts/data/swap_curve_set1.xlsx'

df = pd.read_excel(datapath,
                   index_col=0,
                   header=[0,1,2])

df.index = pd.to_datetime(df.index,format='%Y-%m-%d') 
df = df.sort_index()
df.columns = df.columns.droplevel(2)

# select 2019 USD swap rates
df_2019 = df.loc['2019-01-01':'2019-07-30',['USD']].ffill().dropna()
print(list(df_2019.columns.get_level_values(1)))
# viz
plt.plot(df_2019)
plt.xlabel('Date')
plt.ylabel('Swap rate (%)')
plt.title('USD swap rates from Jan-Jul 2019')
plt.legend(list(df_2019.columns.get_level_values(1)))
plt.show()


