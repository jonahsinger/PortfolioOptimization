import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def main():
    # Define stocks and time period
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=2000)

    # Create DataFrames to store results
    returns_df = pd.DataFrame()
    individual_returns = pd.DataFrame()

    # Fetch and calculate daily returns for each stock
    for stock_symbol in stocks:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change() * 100
        returns_df[stock_symbol] = stock_data['Daily_Return']
        individual_returns[stock_symbol] = stock_data['Daily_Return']

    # Remove rows with NaN values
    returns_df = returns_df.dropna()

    # Calculate expected returns and covariance matrix
    E = np.array(returns_df.mean(axis=0)).reshape(-1, 1)
    cov_matrix = np.array(returns_df.cov())

    # Create matrices for the optimization equation
    ones = np.ones((E.shape[0], 1))
    zeros = np.zeros((2, 2))
    A = np.append(np.append(2 * cov_matrix, E.T, axis=0), ones.T, axis=0)
    temp = np.append(np.append(E, ones, axis=1), zeros, axis=0)
    A = np.append(A, temp, axis=1)

    # Create the b vector
    b = np.vstack([np.zeros((cov_matrix.shape[0], 1)), E[1], [1]])

    # Optimize using matrix algebra
    results = inv(A) @ b

    # Grab first n elements of results because those are the weights
    optimal_weights = results[:cov_matrix.shape[0]]

    # Calculate portfolio returns
    portfolio_returns = returns_df.dot(optimal_weights)

    print("Optimal Weights:")
    for stock_symbol, weight in zip(stocks, optimal_weights):
        print(f"{stock_symbol}: {weight[0]}")

    print("Risks:")
    # Calculate and print the risk of the optimal portfolio
    optimal_portfolio_risk = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
    print(f"Risk of the Optimal Portfolio: {optimal_portfolio_risk[0][0]}")

    # Calculate and print the risk of each individual stock
    for stock_symbol in stocks:
        individual_stock_returns = individual_returns[stock_symbol].dropna()
        individual_stock_risk = np.std(individual_stock_returns)
        print(f"Risk of {stock_symbol}: {individual_stock_risk}")

    # Plot portfolio performance
    cumulative_portfolio_returns = (1 + portfolio_returns / 100).cumprod()
    cumulative_individual_returns = (1 + individual_returns / 100).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_portfolio_returns, label='Optimal Portfolio')

    for column in individual_returns.columns:
        plt.plot(cumulative_individual_returns[column], label=column)

    plt.title('Portfolio and Individual Stock Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
