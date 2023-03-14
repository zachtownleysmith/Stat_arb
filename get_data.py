import yfinance as yf
import pandas as pd

tickers = list(pd.read_csv('ticker_list_rev.csv')['Ticker'])
df = yf.download(tickers, start="2018-02-01", end="2023-02-28")['Adj Close']

df.to_csv('sp500_5yr.csv')

djia = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA',
        'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
        'GS', 'HD', 'HON', 'IBM',
        'INTC', 'JNJ', 'JPM', 'MCD', 'MRK',
        'MSFT', 'NKE', 'PG', 'CRM', 'TRV',
        'UNH', 'VS', 'V', 'WMT', 'WBA']

df = yf.download(djia, start="2018-02-01", end="2023-02-28")['Adj Close']
df.to_csv('djia_5yr.csv')

