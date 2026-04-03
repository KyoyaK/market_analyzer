"""
Description: This code implements analyses of 
financial data pulled from yfinance. There are 
two types of stock predictions implemented, 
one predicting returns and the other the closing 
price of the stock. Both of these models use scikit-learn's
linear regression model. The trainig data used was a certain
fraction of the financial data (default set of 0.8), 
and the testing data was the remaining fraction. 
The accuracy of the models may be limited due to a number of 
reasons, such as the simplicity of the model, 
the 'noisyness' of the stock market, and more. 

Bugs: None known. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class StockAnalyzer:
    def __init__(self, stock):
        self.stock = stock
        self.ticker = yf.Ticker(stock)
        self.financials = self.ticker.financials
        self.history = self.ticker.history("1y")
        self.close = self.history['Close']
        self.open = self.history['Open']
    
    def __str__(self):
        return self.stock

    
    def volatility(self):
        return self.returns().rolling(20).std() 
    
    def simple_moving_average(self, past_days=20):
        return self.close.rolling(past_days).mean()
    
    def corr(self, others):
        matrix = self.close.rename(self.stock).copy(deep=True)
        for stock in others:
            matrix = pd.concat((matrix, stock.close.rename(stock.stock)), axis=1)
        return matrix.corr()
    
    def returns(self):
        return self.close.pct_change()
    
    def predict_returns(self, train_split=0.8, plot=False):
        """Predict next-day returns using linear regression. Change variable train_split
        to change the fraction of data used as training data"""
        df = pd.DataFrame({
            "returns" : self.returns(),
            "sma_20" : self.simple_moving_average(), 
            "sma_5" : self.simple_moving_average(5),
            "volatility" : self.volatility(), 
            "target" : self.returns().shift(-1)
        })

        df = (df - df.mean())/df.std()
        df.dropna(inplace=True)

        training_data = int(train_split*len(df))

        training_x = df[["returns", "sma_20", "sma_5", "volatility"]][:training_data]
        training_y = df["target"][:training_data]
        model = LinearRegression()
        model.fit(training_x, training_y)

        test_x = df[["returns", "sma_20", "sma_5", "volatility"]][training_data:]
        test_y = df["target"][training_data:]
        prediction = model.predict(test_x)
        error = mean_squared_error(test_y, prediction)
        print(dict(zip(training_x.columns, model.coef_)))

        if plot:
            fig, ax = plt.subplots()
            ax.set_title(f"{self.stock}\nPrediction vs. Real Data: \nMSE:{error}")
            ax.plot(np.arange(len(df)), np.hstack((training_y, prediction)),
                    c="r", label="Prediction")
            ax.plot(np.arange(len(df)), df["target"], label="Real data")
            ax.legend()
            plt.show()
            

        return prediction, error
    
    def predict_close(self, train_split=0.8, plot=False):
        """Predict closing prices 1 day ahead. Change variable train_split
        to change the fraction of data used as training data"""
        df = pd.DataFrame({
            "sma_20" : self.simple_moving_average(), 
            "sma_5" : self.simple_moving_average(5),
            "volatility" : self.volatility(), 
            "target" : self.close.shift(-1)
        })
        

        df = (df - df.mean())/df.std()
        df.dropna(inplace=True)

        training_data = int(train_split*len(df))

        training_x = df[["sma_20", "sma_5", "volatility"]][:training_data]
        training_y = df["target"][:training_data]
        model = LinearRegression()
        model.fit(training_x, training_y)

        test_x = df[["sma_20", "sma_5", "volatility"]][training_data:]
        test_y = df["target"][training_data:]
        prediction = model.predict(test_x)
        error = mean_squared_error(test_y, prediction)
        print(dict(zip(training_x.columns, model.coef_)))

        if plot:
            fig, ax = plt.subplots()
            ax.set_title(f"{self.stock}\nPrediction vs. Real Data: \nMSE:{error:.2}")
            ax.plot(np.arange(len(df)), np.hstack((training_y, prediction)),
                    c="r", label="Prediction")
            ax.plot(np.arange(len(df)), df["target"], label="Real data")
            ax.legend()
            plt.show()

        return prediction, error

if __name__ == "__main__":

    #CORRELATION MATRIX

    aapl = StockAnalyzer("AAPL")
    goog = StockAnalyzer("GOOG")
    amd = StockAnalyzer("AMD")
    nvda = StockAnalyzer("NVDA")
    print(aapl.corr((goog, amd, nvda)))

    fig, ax = plt.subplots()
    aapl.close.plot(label = "Apple")
    goog.close.plot(label = "Google")
    amd.close.plot(label="AMD")
    nvda.close.plot(label="NVidia")
    ax.legend()

    plt.show()

    #PREDICTIONS

    aapl.predict_close(plot=True)
    goog.predict_close(plot=True)


