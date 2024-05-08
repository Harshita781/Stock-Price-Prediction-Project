import streamlit as st
import yfinance as yf
import pandas as pd
import yfinance as yf

def loading(symbol):
    stock = yf.Ticker(symbol)
    stock_hist = stock.history(start="2001-01-01")

    data = stock_hist[["Close"]]
    data = data.rename(columns = {'Close': 'Actual_Close'})
    data["Target"]= stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    
    if data.empty:
        st.error("Invalid Stock Symbol")
    
    stock_prev = stock_hist.copy()

    stock_prev = stock_prev.shift(1)

    predictors = ["Close", "High", "Low", "Open", "Volume"]
    data = data.join(stock_prev[predictors]).iloc[1:]
    return data

def model(symbol,start,step,data):                                                   
    

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

    import pandas as pd

    def backtest(data, model, predictors, start, step):
        predictions = []
        for i in range(start, data.shape[0], step):
        
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            
            model.fit(train[predictors], train["Target"])
            
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index = test.index)
            preds[preds > .5] = 1
            preds[preds <= .5] = 0
            
            combined = pd.concat({"Target": test["Target"], "Predictions":preds}, axis=1)
        
            predictions.append(combined)
        
        predictions = pd.concat(predictions)
        return data, predictions

    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    annual_mean = data.rolling(365).mean()

    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["Close"]/data["Close"]
    data["quarterly_mean"] = quarterly_mean["Close"]/data["Close"]
    data["annual_mean"] = annual_mean["Close"]/data["Close"]

    data["annual_weekly_mean"] = data["annual_mean"]/data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"]/data["quarterly_mean"]
    data["quarterly_weekly_mean"] = data["quarterly_mean"]/data["weekly_mean"]

    predictors = ["Close", "High", "Low", "Open", "Volume","weekly_mean","quarterly_mean","annual_mean","annual_weekly_mean","annual_quarterly_mean","quarterly_weekly_mean"]

    return backtest(data, model, predictors, start, step)

def trading(predictions, data, time_horizon, stop_loss=None):
    profit = 0
    data = data.iloc[100:]
    prediction_dates = predictions.index.tolist()
    
    # Initialize variables to track the last trading date and the last trade price
    last_trading_date = None
    last_trade_price = None
    
    # Adjust data index to match predictions index
    for i, prediction_date in enumerate(prediction_dates):
        if predictions.iloc[i]["Predictions"] == 1:
            # Get the corresponding index from predictions
            prediction_index = predictions.index[i]
            # Use the prediction index to retrieve data
            data_index = data.index[data.index.get_loc(prediction_index)]
            # Initialize trading_date
            trading_date = data_index
            # Determine the trading period based on the time horizon
            if time_horizon == "daily":
                trading_date = prediction_date
            elif time_horizon == "weekly":
                # Get the trading date for the end of the week
                trading_date = prediction_date + pd.DateOffset(days=7)
            elif time_horizon == "monthly":
                # Get the trading date for the end of the month
                trading_date = prediction_date + pd.DateOffset(months=1)
            elif time_horizon == "quarterly":
                # Get the trading date for the end of the quarter
                trading_date = prediction_date + pd.DateOffset(months=3)
            elif time_horizon == "yearly":
                # Get the trading date for the end of the year
                trading_date = prediction_date + pd.DateOffset(years=1)
            
            # Ensure the trading date is within the data range
            if trading_date in data.index:
                # Check if a stop loss is triggered
                if stop_loss is not None and last_trade_price is not None:
                    if data.loc[trading_date, 'Low'] < last_trade_price * (1 - stop_loss):
                        profit += data.loc[trading_date, 'Close'] - last_trade_price
                        last_trade_price = None  # Reset last trade price
                        last_trading_date = None  # Reset last trading date
                        continue  # Skip to the next prediction date
                
                # Execute a trade
                if last_trading_date is None:
                    # Buy if no previous trade
                    last_trade_price = data.loc[trading_date, 'Open']
                    last_trading_date = trading_date
                else:
                    # Sell if a previous trade exists
                    profit += data.loc[trading_date, 'Close'] - last_trade_price
                    last_trade_price = None  # Reset last trade price
                    last_trading_date = None  # Reset last trading date

    # Check for an open position at the end
    if last_trade_price is not None:
        # Sell at the last available price
        profit += data.iloc[-1]['Close'] - last_trade_price
    
    return profit

        
                
def main():
    st.sidebar.image("logo.png", use_column_width=True)

    st.title("Stock Predictor")

    @st.experimental_dialog("How to use")
    def modal_content():
        st.write("1. Click on the top left arrow if you can't see the input fields.")
        st.write("2. Enter the stock symbol of your choice in the text box provided.")
        st.write("3. Select the trading strategy you wish to employ from the dropdown menu.")
        st.write("4. Adjust any advanced settings if needed, such as the start index and step size.")
        st.write("5. Optionally, you can also add a stop loss by entering the desired value in the provided text box.")
        st.write("6. Click the 'Run' button to see the predictions and the actual stock value over time.")
        st.write("Explore the app further to visualize the predictions and analyze the results.")
        st.write("Enjoy predicting stock trends with ease!")

    open_modal_button = st.button("Tip💡")

    if open_modal_button:
        modal_content()

    # User input for stock symbol
    symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)')
    trading_options = ['Daily','Weekly','Monthly','Quarterly','Yearly']
    time_horizon = st.sidebar.selectbox('Enter the trading strategy that you wish to employ', options=trading_options)
    stoploss = st.sidebar.text_input('Enter the stop loss in dollars $')
    try:
        stoploss = float(stoploss) if stoploss else None
    except ValueError:
        st.error("Please enter a valid number for the stop loss.")

    st.sidebar.warning('Leave empty if not required')

    # Advanced settings in the left sidebar
    st.sidebar.subheader('Advanced Settings')
    st.sidebar.warning("For faster results, use high step values")
    st.sidebar.warning(" For accuracte results, use lower step values")
    start = st.sidebar.number_input('Start', value=2500, help="Start index for the prediction range.")
    step = st.sidebar.number_input('Step', value=750, help="Step size for the prediction range.")

    if st.sidebar.button('Run'):
        if symbol:
            # Attempt to retrieve data for the provided stock symbol
            try:
                data = loading(symbol)
                data, predictions = model(symbol, start, step, data)
                profit = trading(predictions, data, time_horizon, stoploss)

                from sklearn.metrics import precision_score 
                precision = 100 * precision_score(predictions["Target"], predictions["Predictions"])

                # Plot the predictions vs target and the actual stock value over time
                import matplotlib.pyplot as plt

                # Create a figure with two subplots side by side, with larger size
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Adjust figsize here

                # Plot the predictions vs target
                ax1.plot(predictions.index, predictions['Target'], label='Target', marker='o')
                ax1.plot(predictions.index, predictions['Predictions'], label='Predictions', marker='x')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Value')
                ax1.set_title('Target vs Predictions')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)

                # Plot the data line
                data["Close"].plot(ax=ax2, use_index=True)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Close Value')
                ax2.set_title('Close Value Over Time')
                ax2.grid(True)

                # Display the plots using Streamlit
                st.pyplot(fig)


                # Display the profit value and accuracy on the main screen
                st.markdown('<h2 style="text-align: center; color: blue;">Model Precision:{:.2f}</h2>'.format(precision), unsafe_allow_html=True)
                if profit > 0:
                    st.markdown('<h2 style="text-align: center; color: green;">Profit: ${:.2f}</h2>'.format(profit), unsafe_allow_html=True)
                    st.toast("The profit was calculated using the greedy algorithm.")
                    st.balloons()
                elif profit == 0:
                    st.markdown('<h2 style="text-align: center; color: gray;">No Profit & No Loss </h2>', unsafe_allow_html=True)
                else:
                    st.markdown('<h2 style="text-align: center; color: red;">Loss: ${:.2f}</h2>'.format(abs(profit)), unsafe_allow_html=True)
                    st.toast("The profit was calculated using the greedy algorithm.")
                
            except Exception as e:
                st.error(e)


if __name__ == '__main__':
    main()
