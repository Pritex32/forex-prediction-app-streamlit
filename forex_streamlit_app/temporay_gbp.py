


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model



import datetime
from datetime import date,timedelta

# title of the app

st.title('Stock Price Prediction app') 
st.subheader('Author: Prisca Ukanwa')

st.write('notification: put in only the ticker ID of the stock')


import time

gbpusd=st.text_input('Enter the stock ID','GBPUSD')

import yfinance as yf

# Fetch GBP/USD forex data with a daily interval
import yfinance as yf

# Fetch GBP/USD forex data from 2010 to present with daily interval
# Function to fetch GBP/USD data with caching
today = date.today().strftime("%Y-%m-%d")
@st.cache_data
def get_forex_data():
    gbpusd = yf.Ticker("GBPUSD=X")
    
    hist = gbpusd.history(start="2010-01-01", end=today, interval="1d")
    return hist

# Load cached data
data = get_forex_data()

#renamimg columns






# to show the data downloaded
st.subheader('stock Data')

st.write(data.head())





model=load_model('C:\\Users\\USER\\Documents\\vscode\\forex_streamlit_app\\gbpusd_stock_model.h5')

st.sidebar.header('App Details')
st.sidebar.write('Welcome to the **Future Price Forecasting App**! This application is designed to assist traders and investors in making informed decisions by providing accurate price forecasts for stocks currency pairs, or other financial instruments.')
st.sidebar.write(""" With this tool, you can:
- **Minimize trading losses** by anticipating market trends.
- **Make strategic investment decisions** with data-driven forecasts.
- **Plan for the future** using customizable prediction periods.""")
st.sidebar.subheader('How to Use the App')
st.sidebar.write("""
1. **Enter the Stock or Pair ID**: Input the identifier for the stock or currency pair you want to forecast.
2. **Select the Start Date**: Choose the starting point for your prediction period.
3. **Set the Prediction Period**: Indicate how many days, months, or years you want to forecast.
4. **Download the Forecast**: After generating the predictions, you can download the forecasted values in a convenient format.
""")

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)



rollmean=data['Open'].rolling(50).mean() # the moving average

# plotting the Open column
st.subheader('Price vs moving Average')
fig=plt.figure(figsize=(15,10))
plt.plot(data['Open'],label='yearly chart')
plt.plot(rollmean,label='50 MA')
plt.legend()
plt.title('Open price yearly chart')
st.pyplot(fig)


# selecting target column
open_price=data[['Open']]



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


scaler=MinMaxScaler(feature_range=(0,1))
# scaling the data
scaler_data=scaler.fit_transform(open_price)
# loading the model

# feature sequences
x=[]
y=[]
for i in range (60,len(scaler_data)):
      x.append(scaler_data[i-60:i])
      y.append(scaler_data[i])
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


pred=model.predict(x_test)

inv_pred=scaler.inverse_transform(pred)

inv_y_test=scaler.inverse_transform(y_test)


#plotting the predicted vs actual values
from statsmodels.tsa .api import SimpleExpSmoothing

st.subheader('Actual values vs model predicted to validate accuracy')
fig9=plt.figure(figsize=(15,10))
fit1=SimpleExpSmoothing(inv_pred).fit(smoothing_level=0.02,optimized=False)
plt.plot(fit1.fittedvalues,color='red',label='Predicted values')
fit2=SimpleExpSmoothing(inv_y_test).fit(smoothing_level=0.02,optimized=False)
plt.plot(fit2.fittedvalues,color='blue',label='Actual')
plt.legend()
st.pyplot(fig9)

from sklearn.metrics import mean_squared_error
# the mean squared error
rmse=np.sqrt(mean_squared_error(inv_y_test,inv_pred))
st.write('mean_squared_error',rmse)





st.subheader('Forecast')
st.write('select date range')
# putting the date range for future prediction



n_period = st.number_input(label='How many days of forecast?', min_value=1, value=30)



future_days = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_period,freq='D')


        
    
# values to predict 

last_days=scaler_data[-60:]
last_days=last_days.reshape(1,60,1)
future_prediction=[]
for _ in range(n_period):
    nxt_pred=model.predict(last_days)
    future_prediction.append(nxt_pred[0,0])
    last_days = np.append(last_days[:, 1:, :], [[[nxt_pred[0, 0]]]], axis=1)
    last_days[0,-1,0]=nxt_pred
    
forecast_array=np.array(future_prediction)
future_prediction=scaler.inverse_transform(forecast_array.reshape(-1,1))

st.write('prediction', nxt_pred)
# wether scaled values or x_test,it still the same thing

# plotting monthly chart



# dataframe for the forecated values
f_df = pd.DataFrame({'dates':future_days,
                     'open':future_prediction.flatten()}) 

f_df['dates']=pd.to_datetime(f_df['dates'])    

st.subheader('Forecasted values')
st.write('Click the Arrow ontop of this table to download this predictions')
st.dataframe(f_df)

st.subheader(' Daily forecasted value chart')
fig2=plt.figure(figsize=(15,10))
plt.plot(f_df['dates'],f_df['open'],label='forecasted values',marker='o',markerfacecolor='yellow',color='red')
plt.xticks(rotation='vertical')
plt.title('predictions')
plt.grid(True)
plt.legend()
plt.show()
st.pyplot(fig2)



df=data.copy()
df.index = pd.to_datetime(df.index)
# Resample Open price
monthly_open = df['Open'].resample('M').mean()  # Monthly average
weekly_open = df['Open'].resample('W').mean()   # Weekly average



st.subheader('monthly chart')
fig0=plt.figure(figsize=(15,10))
plt.plot(monthly_open )
plt.title('monthly data')
st.pyplot(fig0)

st.subheader('Weekly chart')
fig11=plt.figure(figsize=(15,10))
plt.plot(weekly_open,marker='D',markerfacecolor='yellow')
plt.title('monthly data')
st.pyplot(fig0)




# this chart shows that in the next 6months prices will range at 1.7 price in gbpusd
st.subheader('Forecasted value and original data')


# plotting the entire df with the forecated values
fig3=plt.figure(figsize=(20,15))
sns.lineplot(x=data.index,y=data['Open'],label='original values')
sns.lineplot(x=f_df['dates'],y=f_df['open'],label='forecasted values',color='red')
plt.title('Days of predictions, full view')
plt.grid(True)
st. pyplot(fig3)



# Ensure date input is valid before filtering
# selecting the open column from the main df

# plotting the entire df by filtering  to get a close view


