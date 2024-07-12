# import required libraries
import pandas as pd
import numpy as np
import yfinance as yf 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")
from pmdarima import auto_arima


# first checkpoint 
print("Libraries imported successfully.")

# data wrangling
def data_wrangle(path, droped_columns=['Adj Close', 'Volume']):
    """ A method that will clean the original dataset, 
        restructure the dataset and fill the missing values.
        
        input
        -----
        path: data path 
        dropped_columns: columns to be dropped"""
    
    # read the dataset through the path
    df=pd.read_csv(path)
    # change the "Date" column to datetime data type
    df['Date']=pd.to_datetime(df['Date'])
    # set the "Date" column to index
    df=df.set_index('Date')
    # assigned the desired frequecy to set up
    # 'D' stands for day
    desired_frequency = 'D'
    # set the frequency 
    df = df.asfreq(desired_frequency)
    # drop the unnecessary columns that are already specified 
    df = df.drop(columns=droped_columns)
    # fill the missing values 
    df=df.fillna(method='ffill')
    # return the dataframe 
    return df

df_one_year = data_wrangle("data/gold_one_year.csv")

# second checkpoint 
print("Data preprocessing is done.")

def find_parameters(dataframe, target_cloumn):
    """ A method that search for the best p, d and q values for the arima model."""
    stepwise_model = auto_arima(dataframe[target_cloumn], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False,
                            d=1, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=True)

    print(stepwise_model.summary())

# print(df_one_year.head(5))
def arima_forecast(dataframe, target_variable, decomposition_model, future_days):
    """An arima method that train the trend to get the future market prices."""

    # decompose first to keep the trend
    decomposition = seasonal_decompose(dataframe[target_variable], decomposition_model)

    # get the trend without null values
    trend = decomposition.trend.dropna()

    # train/test splitting 
    # splitting the trend data into the training set and the test set
    train=trend[:int(0.80*len(trend))] # train data 80 % 
    #print(int(0.8*len(trend)))
    #print(len(trend))
    test=trend[int(0.80*len(trend)):] # test data 20 %

    # Creating a list to store future dates
    last_date = trend.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days + 1)
    # days to forecast 
    future_days = future_days

    # forecasting with walk-forward validation
    # starts an empty series to store the predicted values
    prediction = pd.Series() 
    # training set starts with train, and gradually increases by 1 observation with each passing day.
    history = train.copy() 

    for i in range(1,1+len(test)+future_days):
        # model is trained on history which increases with each loop
        ARIMA_Model = ARIMA(history,order=(4,1,1)).fit() 
        # gives the prediction for the next timestamp
        next_prediction=ARIMA_Model.forecast()  
        # setting the index for the next prediction
        if i <= len(test):
            next_date = trend.index[len(train) + i - 1]
        else:
            next_date = future_dates[i - len(test) - 1]
        # add to the prediction data frame
        next_prediction.index = [next_date]
    
        # puts all the predictions and timestamps into the series prediction
        prediction = pd.concat([prediction, next_prediction])  
    
        if i <= len(test):
            # continue updating the history with actual test data
            history = trend[:len(train) + i]  # Training set increases by one observation in preparation for the next loop
        else:
            # after test data, keep updating history with predictions
            history = pd.concat([history, next_prediction])
    
    # visualization 
    plt.figure(figsize=(16,8))
    plt.plot(test, label="Test data")
    plt.plot(prediction, label="Forecast")
    plt.legend()
    plt.show()
        
    # return the predicted data
    return prediction




# fourth checkpoint 
print("Successfully implemented ARIMA model.")

# fifth checkpoint 
print("Training data.")

prediction = arima_forecast(df_one_year, 'Close', 'additive', 10)

print("The future market will be.")
print(prediction[-10:])
