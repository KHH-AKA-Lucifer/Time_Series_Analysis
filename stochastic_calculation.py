# import required libraries 
import os
import time 
import warnings 
import numpy as np
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt
from trading_ig import IGService
warnings.filterwarnings('ignore')
from trading_ig.config import config

class slowDcalculation:
    def __init__(self, epic: str):
        self.epic = epic
        self.ig_service = None
    
    def set_connection(self):
        """
        A method that sets up connection to IG API.
        """
        # initialize IG service
        self.ig_service = IGService(
                        config.username,
                        config.password,
                        config.api_key,
                        config.acc_type
                    )

        # login 
        self.ig_service.create_session()
        print("Connected successfully!")

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame to have separate columns for bid, ask, and last data types.

        Parameters
        ----------
        df: dataframe 

        Returns
        -------
        transformed_df: dataframe which is already transformed 
        """
        # define the new columns for the transformed DataFrame
        columns = ['bid_Open', 'bid_High', 'bid_Low', 'bid_Close',
                   'ask_Open', 'ask_High', 'ask_Low', 'ask_Close',
                   'last_Open', 'last_High', 'last_Low', 'last_Close',
                   'Volume']

        # create a new DataFrame with the desired columns
        transformed_df = pd.DataFrame(index=df.index)

        # extract bid data
        transformed_df['Open'] = df[('bid', 'Open')]
        transformed_df['High'] = df[('bid', 'High')]
        transformed_df['Low'] = df[('bid', 'Low')]
        transformed_df['Close'] = df[('bid', 'Close')]
     
        # extract volume
        transformed_df['Volume'] = df[('last', 'Volume')]

        # return the transofrmed dataframe 
        return transformed_df
    
    def fetch_gold_data(self, existing_csv_path, resolution, date_column = 'Date'):
        """
        Fetch historical gold data from IG with specified resolution and date range.
    
        Parameters
        ----------
        ig_service (IGService): The initialized IGService object.
        resolution (str): The data resolution (e.g., '1Min', '1H', '1D').
        from_date (str): The start date for data retrieval in ISO 8601 format.
        numpoints (int): The maximum number of data points to retrieve.
    
        Returns
        -------
        pandas.DataFrame: A DataFrame containing the historical data.
        """
        # read existing csv_path to determin start date / time
        existing_data = pd.read_csv(existing_csv_path, parse_dates=[date_column], index_col=date_column)

        # convert to processable string format
        from_date = existing_data.index[-1].strftime('%Y-%m-%dT%H:%M:%S')
        
        # define the epic for gold
        # epic = "CS.D.USCGC.TODAY.IP"  # Example epic for gold (Check IG API for correct epic)

        # get the current date and time
        to_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # request historical data
        response = self.ig_service.fetch_historical_prices_by_epic(
            self.epic,
            resolution=resolution,
            start_date=from_date,
            end_date=to_date,
        )

        # convert the response to a DataFrame
        df = response['prices']
        transformed_df = self.transform_dataframe(df)
    
        # retrun the transformed dataframe 
        return transformed_df

    def update_gold_data(self, existing_csv_path, new_data, date_column='Date'):
        """
        Updates the existing gold data CSV file with new data retrieved from the IG API.
    
        Parameters
        ----------
        existing_csv_path: Path to the existing 1-hour CSV file
        new_data: The new data DataFrame retrieved from the IG API
        date_column: The name of the date column (default is 'Date')

        Returns
        -------
        combined_df: combined dataset (old data + new data)
        """
        # load the existing 1-hour data CSV file
        existing_data = pd.read_csv(existing_csv_path, parse_dates=[date_column], index_col=date_column)
    
        # convert the 'DateTime' to datetime and set it as the index
        new_data['Date'] = pd.to_datetime(new_data.index)
        new_data.set_index('Date', inplace=True)
    
        # drop any duplicate index entries
        new_data = new_data[~new_data.index.duplicated(keep='first')]
    
        # merge the dataframes
        # Concatenate the new data with the existing data
        combined_data = pd.concat([existing_data, new_data])
    
        # drop any duplicate rows that may exist after concatenation
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
    
        # sort the combined data by date to maintain chronological order
        combined_data.sort_index(inplace=True)
    
        # save the Updated CSV
        # Save the combined data back to the CSV file
        # combined_data.to_csv(existing_csv_path)
        return combined_data

    def update_every_minute(self, csv_path, resolution='1Min'):
        """
        A method to update the gold data every minute.

        Parameters
        ----------
        csv_path : str
            Path to the existing CSV file where data is stored.
        resolution : str, optional
            Data resolution for fetching new data (default is '1Min').
        """
        # self.set_connection()  # Ensure the IG connection is active
        
        while True:
            # Fetch the latest data from IG API
            new_data = self.fetch_gold_data(existing_csv_path=csv_path, resolution=resolution)
            
            # Update the existing CSV file with the new data
            updated_data = self.update_gold_data(existing_csv_path=csv_path, new_data=new_data)
            
             # Save the updated data to the CSV file after each update
            #updated_data.to_csv(csv_path)
            #print("Data saved successfully after 20 minutes.")
            
            # Wait for 20 minutes before the next update
            #time.sleep(1200)

            # Save the updated data back to the CSV file
            updated_data.to_csv(csv_path)
            print("Data updated successfully.")
            
            # Wait for 1 minute before the next update
            time.sleep(300)
    
    def stochastic(self, df: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> pd.DataFrame:
        """A stochastic function that calculates the Fast %K & Slow %D using EMA.
    
        Parameters
        ----------
        df: pd.DataFrame (Input dataframe containing OHLC data.)
        k_period: int, optional (Period to calculate the Fast %K <default is 9>.)
        d_period: int, optional (Period to calculate the Slow %D <default is 3>.)
    
        Returns
        -------
        pd.DataFrame (DataFrame that contains Fast %K, Fast %D (EMA), and Slow %D (EMA).)
        """

        # find the highest high market price in the k period
        df['HighestHigh'] = df['High'].rolling(window=k_period).max()

        # find the lowest low market price in the k period
        df['LowestLow'] = df['Low'].rolling(window=k_period).min()

        # calculate Fast %K
        df['FastK'] = ((df['Close'] - df['LowestLow']) / (df['HighestHigh'] - df['LowestLow'])) * 100

        # calculate Fast %D (EMA of Fast %K with period 1, which is just FastK itself)
        df['FastD'] = df['FastK']

        # calculate Slow %D (EMA of Fast %D with period d_period)
        df['SlowD'] = df['FastD'].ewm(span=d_period, adjust=False).mean()

        # drop temporary columns
        df.drop(columns=['HighestHigh', 'LowestLow'], inplace=True)

        # Return the dataframe with stochastic values
        return df

    def resample_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Resample the dataframe to a specified time interval.

        Parameters
        ----------
        df : pd.DataFrame
        Input dataframe with datetime index.
        interval : str
            Time interval for resampling (e.g., '3h', '4h', '6h', '9h').

        Returns
        -------
        pd.DataFrame
        Resampled dataframe with Open, High, Low, Close, and Volume columns.
        """

        # ensure 'Date' is a datetime and set it as the index
        df.index = pd.to_datetime(df.index)
    
        # resample the dataset to the specified interval
        df_resampled = df.resample(interval).agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    })

        # drop missing values
        df_resampled.dropna(inplace=True)

        return df_resampled
    
    def calculate_and_print_slow_d(self, df: pd.DataFrame, label: str) -> float:
        """ a method to calculate stochastic indicators and print the latest slow %D value
        
        parameter
        ---------
        df: input dataframe
        label: label for the output (e.g., 'live', '1hr', '3hr', '6hr')"""
        stochastic_df = self.stochastic(df)
        latest_slow_d = stochastic_df['SlowD'].iloc[-1]
        print(f'{label} Slow %D: {round(latest_slow_d,4)}')
        return stochastic_df

    def compare(self, current_time: str, current_slowd: float, prev_time: str, prev_slowd: float):
        """
        a method that compares two slow d from different time frames 
        (e.g. live, 1hr, 3hr, 6hr, 1day)

        parameter
        ---------
        current_time: current time frame (a day, an hour, etc)
        current_slowd: slow d of current time frame
        prev_time: previous time frame (a day, an hour, etc) 
        prev_slowd: slow d of prev time frame

        return
        ------
        None
        string output: up/ down
        """
        print(f'{current_time} Slow D% is {current_slowd:.4f} and {prev_time} Slow D% is {prev_slowd:.4f}.')
        if current_slowd > prev_slowd:
            print(f"{current_time} Slow D% is greater than {prev_time} Slow D% and the market bias is up.")
        elif current_slowd < prev_slowd:
            print(f"{current_time} Slow D% is less than {prev_time} Slow D% and the market bias is down.")
        else:
            print("stable")


""" # instantiate the class
slowd = slowDcalculation(epic='CS.D.USCGC.TODAY.IP')

slowd.set_connection()

# fetch data from IG API
root_path = os.getcwd()
# 1 min data file path (for resample)
existing_1min_data_path = os.path.join(root_path, 'data\gold_minutely_data.csv')
# 1 hour data file path
existing_1hr_data_path = os.path.join(root_path, 'data\gold_hourly_data.csv')
# 1 day data file path 
existing_1d_data_path = os.path.join(root_path, 'data\gold_daily_data.csv')

transformed_1m_df = slowd.fetch_gold_data(existing_1min_data_path, resolution="1Min", date_column = 'Date')
# print(transformed_1m_df)

transformed_1hr_df = slowd.fetch_gold_data(existing_1hr_data_path, resolution="1h", date_column = 'Date')
# print(transformed_1hr_df)

transformed_1d_df = slowd.fetch_gold_data(existing_1d_data_path, resolution="1D", date_column = 'Date')
# print(transformed_1d_df)

# update dataset 
updated_1m_df = slowd.update_gold_data(existing_1min_data_path, transformed_1m_df)
# print(updated_1m_df.tail(5))
# save to the file 
updated_1m_df.to_csv(existing_1min_data_path)

updated_1hr_df = slowd.update_gold_data(existing_1hr_data_path, transformed_1hr_df)
# print(updated_1hr_df.tail(5))
# save to the file 
updated_1hr_df.to_csv(existing_1hr_data_path)

updated_1d_df = slowd.update_gold_data(existing_1d_data_path, transformed_1d_df)
# print(updated_1d_df.tail(5))
# save to the file 
updated_1d_df.to_csv(existing_1d_data_path)

slowd.update_every_minute(csv_path=existing_1min_data_path) """


""" # load the data back 
minutely_data = pd.read_csv(existing_1min_data_path, parse_dates=['Date'], index_col='Date')
hourly_data = pd.read_csv(existing_1hr_data_path, parse_dates=['Date'], index_col='Date')
daily_data = pd.read_csv(existing_1d_data_path, parse_dates=['Date'], index_col='Date')

minutely_slowd = slowd.stochastic(minutely_data)
minutely_data.to_csv('data/minutely_slowd.csv')
hourly_slowd = slowd.stochastic(hourly_data)
hourly_data.to_csv('data/hourly_slowd.csv')
daily_slowd = slowd.stochastic(daily_data)
daily_slowd.to_csv('data/daily_slowd.csv')

print(minutely_data.tail(5))
print('-'*100)
print(hourly_data.tail(5))
print('-'*100)
print(daily_data.tail(5))

# check 4-hour resample 
df_4h = slowd.resample_data(minutely_data, '4h')
# calculate slowd of 4-hour resample 
df_4h_stochastic = slowd.stochastic(df_4h)
# print(df_4h_stochastic.tail(5))
assert round(df_4h_stochastic.loc['2024-08-14 20:00:00']['SlowD'],4) == 27.2868
print('Test passed')

# resample 3, 6, 9 hours
df_3h =  slowd.resample_data(minutely_data, '3h')
df_6h = slowd.resample_data(minutely_data, '6h')
df_9h = slowd.resample_data(minutely_data, '9h')



# calculate and print slow %D values
live_slow_d = slowd.calculate_and_print_slow_d(minutely_data, 'live')
hourly_slow_d = slowd.calculate_and_print_slow_d(hourly_data, '1hr')
daily_slow_d = slowd.calculate_and_print_slow_d(daily_data, '1-day')
slow_d_4h = slowd.calculate_and_print_slow_d(df_4h, '4hr')
slow_d_3h = slowd.calculate_and_print_slow_d(df_3h, '3hr')
slow_d_6h = slowd.calculate_and_print_slow_d(df_6h, '6hr')


# compare current slow d and yesterday slow d
slowd.compare("Live", live_slow_d['SlowD'].iloc[-1], "Yesterday", daily_slow_d['SlowD'].iloc[-1])
# compare current slow d and previous 3 hour slow d
slowd.compare("Live", live_slow_d['SlowD'].iloc[-1], "Previous 3 hour", slow_d_3h['SlowD'].iloc[-1])
# compare current slow d and previous 6 hour slow d
slowd.compare("Live", live_slow_d['SlowD'].iloc[-1], "Previous 6 hour", slow_d_6h['SlowD'].iloc[-1])
# compare current 4 hour slow d and previous 4 hour slow d
slowd.compare("Current 4 hour", slow_d_4h['SlowD'].iloc[-1], "Previous 4 hour", slow_d_4h['SlowD'].iloc[-2])
# compare current 3 hour slow d and previous 3 hour slow d
slowd.compare("Current 3 hour", slow_d_3h['SlowD'].iloc[-1], "Previous 3 hour", slow_d_3h['SlowD'].iloc[-2])
# compare current 6 hour slow d and previous 6 hour slow d
slowd.compare("Current 6 hour", slow_d_6h['SlowD'].iloc[-1], "Previous 6 hour", slow_d_6h['SlowD'].iloc[-2])
# compare yesterday slow d and the day before yesterday  slow d
slowd.compare("Yesterday", daily_slow_d['SlowD'].iloc[-2], "The day before yesterday", daily_slow_d['SlowD'].iloc[-3]) """