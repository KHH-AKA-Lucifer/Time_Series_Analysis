# data collection 

# import y_finance library 
import yfinance as yf 

# one day - 1 min interval
gold_one_day = yf.download('GC=F', interval="1m")
# one month - 5 min interval
gold_one_month = yf.download('GC=F', interval="5m", period="1mo")
# one year - 1 day interval
gold_one_year = yf.download('GC=F', period="1y")
# ten years - 1 day interval 
gold_ten_year = yf.download('GC=F', period="10y")

# check for missing values
for index, i in {"gold_one_day": gold_one_day, "gold_one_month": gold_one_month, 
                 "gold_one_year": gold_one_year, "gold_one_year": gold_one_year, "gold_ten_year": gold_ten_year}.items():
    print(index)
    print("------------")
    print(i.isna().sum(), end="\n\n")


# save the data as csv file
gold_one_day.to_csv('data/gold_one_day.csv')
gold_one_month.to_csv('data/gold_one_month.csv')
gold_one_year.to_csv('data/gold_one_year.csv')
gold_ten_year.to_csv('data/gold_ten_year.csv')