# import required libraries 
import os
import warnings 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Function to calculate CCI
def calculate_cci(data, period):

    # calculate the typical price
    data['Typical Price'] = (data['High'] + data['Low'] + data['Close']) / 3

    # calculate the simple moving average (SMA) of the Typical Price
    sma = data['Typical Price'].rolling(window=period).mean()

    # Calculate the mean deviation manually
    mean_deviation = data['Typical Price'].rolling(window=period).apply(
        lambda x: (np.abs(x - x.mean()).mean()), raw=True
    )



    # calculate the CCI
    cci = (data['Typical Price'] - sma) / (0.015 * mean_deviation)
    
    return cci


# load data set 
hourly_data = pd.read_csv("data/gold_hourly_data.csv")

# calculate CCI
hourly_data['CCI3'] = calculate_cci(hourly_data, 3)
hourly_data['CCI9'] = calculate_cci(hourly_data, 9)

print(hourly_data.tail(5))