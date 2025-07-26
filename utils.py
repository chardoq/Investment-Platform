import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_financial_metrics(prices):
    """
    Calculate comprehensive financial metrics for a stock.
    """
    if len(prices) < 2:
        return {}
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Basic metrics
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(prices)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk metrics
    max_drawdown = calculate_max_drawdown(prices)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Performance metrics
    sharpe_ratio = (annualized_return - 0.045) / volatility if volatility > 0 else 0
    sortino_ratio = calculate_sortino_ratio(returns)
    
    return {
        'Total Return (%)': round(total_return * 100, 2),
        'Annualized Return (%)': round(annualized_return * 100, 2),
        'Volatility (%)': round(volatility * 100, 2),
        'Sharpe Ratio': round(sharpe_ratio, 3),
        'Sortino Ratio': round(sortino_ratio, 3),
        'Max Drawdown (%)': round(max_drawdown * 100, 2),
        'VaR 95% (%)': round(var_95 * 100, 2),
        'CVaR 95% (%)': round(cvar_95 * 100, 2)
    }

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from peak to trough.
    """
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def calculate_sortino_ratio(returns, risk_free_rate=0.045):
    """
    Calculate Sortino ratio using downside deviation.
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.inf
    
    return (returns.mean() * 252 - risk_free_rate) / downside_deviation

def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta coefficient relative to market.
    """
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0
    
    return covariance / market_variance

def get_stock_info(symbol):
    """
    Get comprehensive stock information from yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        return {
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': format_large_number(info.get('marketCap', 0)),
            'P/E Ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
            'PEG Ratio': round(info.get('pegRatio', 0), 2) if info.get('pegRatio') else 'N/A',
            'Dividend Yield (%)': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
            '52W High': round(info.get('fiftyTwoWeekHigh', 0), 2),
            '52W Low': round(info.get('fiftyTwoWeekLow', 0), 2),
            'Current Price': round(info.get('currentPrice', 0), 2)
        }
    except Exception as e:
        return {'Error': f'Could not fetch data for {symbol}'}

def format_large_number(num):
    """
    Format large numbers with appropriate suffixes.
    """
    if num >= 1e12:
        return f'${num/1e12:.2f}T'
    elif num >= 1e9:
        return f'${num/1e9:.2f}B'
    elif num >= 1e6:
        return f'${num/1e6:.2f}M'
    elif num >= 1e3:
        return f'${num/1e3:.2f}K'
    else:
        return f'${num:.2f}'

def calculate_portfolio_metrics(weights, returns, risk_free_rate=0.045):
    """
    Calculate portfolio metrics given weights and asset returns.
    """
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'Expected Return': portfolio_return,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio
    }

def generate_efficient_frontier(returns, num_portfolios=10000):
    """
    Generate efficient frontier for portfolio optimization.
    """
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    np.random.seed(42)
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - 0.045) / portfolio_volatility
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
    
    return results

def get_optimal_portfolio(returns, method='sharpe'):
    """
    Find optimal portfolio weights using different optimization methods.
    """
    from scipy.optimize import minimize
    
    num_assets = len(returns.columns)
    
    def portfolio_metrics(weights):
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_return, portfolio_volatility
    
    def negative_sharpe(weights):
        p_return, p_volatility = portfolio_metrics(weights)
        return -(p_return - 0.045) / p_volatility
    
    def portfolio_volatility(weights):
        return portfolio_metrics(weights)[1]
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess
    initial_guess = num_assets * [1. / num_assets]
    
    if method == 'sharpe':
        result = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'min_volatility':
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x if result.success else initial_guess