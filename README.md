# market_analyzer

This project analyzes stock market data pulled from yfinance and builds a simple predictive  model using historical price data. Features include computations of daily market returns, simple moving averages, rolliung volatility, correlation analysis, as well as predictions and visualizations of next-day returns and closing prices.  

Technology used: 
Python
Numpy
Matplotlib
scikit-learn

Example usage:
from market_analyzer import StockAnalyzer
aapl = StockAnalyzer("AAPL")
aapl.predict_close(plot=True)

Notes:
Model performance is limited due to market noise and simplicity of model

Known Bugs:
None
