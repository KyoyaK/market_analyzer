# market_analyzer

This code implements analyses of financial data pulled from yfinance. There are 
two types of stock predictions implemented, one predicting returns and the other the closing 
price of the stock. Both of these models use scikit-learn'slinear regression model. The trainig data used was a certain
fraction of the financial data (default set of 0.8), and the testing data was the remaining fraction. 
The accuracy of the models may be limited due to a number of reasons, such as the simplicity of the model, 
the 'noisyness' of the stock market, and more. 

Technology used: 
Python, 
Numpy, 
Matplotlib, 
scikit-learn

Example usage:

    from market_analyzer import StockAnalyzer

    aapl = StockAnalyzer("AAPL")

    aapl.predict_close(plot=True)


Notes:

Model performance is limited due to market noise and simplicity of model

Known Bugs:
None
