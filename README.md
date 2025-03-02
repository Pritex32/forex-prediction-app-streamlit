# forex-prediction-app-streamlit

# Detailed Summary of the Stock Price Prediction App
- This Stock Price Prediction App, built using Streamlit, allows users to forecast stock prices based on historical data using a deep learning model. The app fetches real-time stock/forex data, performs analysis, and provides future price predictions using an LSTM model.

# 1. Overview of the App
- Title: Stock Price Prediction App
- Author: Prisca Ukanwa
- Purpose: To help traders and investors make informed decisions by forecasting stock/forex prices.

# 2. Data Retrieval and Preprocessing
- Libraries Used: yfinance, pandas, numpy, matplotlib, seaborn, sklearn, tensorflow/keras

- The app uses Yahoo Finance (yfinance) to retrieve historical GBP/USD forex data from 2010 to the present at a daily interval.
- Data is cached to improve performance.
- Fetches GBP/USD forex data for the last 60 days at an hourly interval.
# 3.Data Cleaning
- Drops missing values (NaN) and duplicates to ensure data quality.
# 4. Data Visualization
- Displays the first few rows of both the daily and hourly datasets.
- Plots Moving Average vs. Open Price
- Uses a 50-day moving average to smooth the time series and identify trends.
- Plots Actual vs. Predicted Values
- Compares model predictions with real prices to validate accuracy.
- Plots Weekly and Monthly Trends
- Uses resampling techniques to create weekly and monthly averages for better trend analysis.
# 5. Model Training and Prediction
Preprocessing for the Model
- Uses MinMaxScaler to scale the Open price data between 0 and 1.
- Creates sequences of 60 past days as input and next day’s price as the target.
- Splits the data into train (70%) and test (30%) sets.
- Loading and Using Pretrained LSTM Model
- Model Path: 'gbpusd_stock_model.h5'
Prediction Process:
- Loads the pre-trained LSTM model to predict stock prices.
- Performs inverse transformation to get actual values.
- Performance Evaluation
- Uses Root Mean Squared Error (RMSE) to evaluate the model’s accuracy.
 # 6. Future Price Forecasting
- Users enter a start date and forecast period (number of days).
- Predicts future prices using the trained LSTM model.
- Generates a dataframe of forecasted values with corresponding dates.
- Plots future predictions to visualize trends.
# 7. Hourly Forecasting
- Loads another LSTM model ('hourly_model_forex.h5') for hourly predictions.
- Allows users to input the number of hours to predict.
- Uses a similar sequence-based approach as the daily model.
- Plots the hourly forecasted values.
# 8. Final Analysis
- Compares forecasted values with original data to assess model reliability.
- Provides insights on expected price movements in the next 6 months.
- Uses SimpleExpSmoothing from statsmodels to smooth predictions.
# 9. User Interface Features
- Sidebar Information
App Details: Explains the purpose of the app.
- How to Use the App:
- Enter the stock/forex ticker.
- Choose a start date.
- Select forecast duration.
- Download forecasted results.
# Interactivity
- Users input stock IDs and forecast duration.
- Downloadable predicted stock prices in table format.
- Interactive charts for daily, weekly, and monthly trends.
