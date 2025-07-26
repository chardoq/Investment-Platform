import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    """
    Robust stock data fetcher with error handling and caching
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_stock_data(self, symbol, period='1y', interval='1d', max_retries=3):
        """
        Fetch stock data with retries and caching
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_timeout:
                return cached_data
                
        # Try to fetch data with retries
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Cache the successful result
                    self.cache[cache_key] = (data, time.time())
                    return data
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        return pd.DataFrame()
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer on exception
                    continue
                else:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
                    return pd.DataFrame()
                    
        return pd.DataFrame()
    
    def get_company_info(self, symbol, max_retries=2):
        """
        Get company information with error handling
        """
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Check if we got valid info
                if info and 'symbol' in info:
                    return info
                elif attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {}
                    
        return {}
    
    def get_multiple_stocks(self, symbols, period='1y'):
        """
        Fetch data for multiple stocks efficiently
        """
        results = {}
        valid_symbols = []
        
        # First, validate symbols
        for symbol in symbols:
            data = self.get_stock_data(symbol, period='5d')  # Quick check
            if not data.empty:
                valid_symbols.append(symbol)
            else:
                st.warning(f"Could not fetch data for {symbol}")
        
        # Fetch full data for valid symbols
        if valid_symbols:
            try:
                # Use yfinance's download function for multiple symbols
                data = yf.download(valid_symbols, period=period, group_by='ticker')
                
                if len(valid_symbols) == 1:
                    results[valid_symbols[0]] = data
                else:
                    for symbol in valid_symbols:
                        if symbol in data.columns.levels[0]:
                            results[symbol] = data[symbol]
                        
            except Exception as e:
                # Fallback to individual fetching
                st.warning("Bulk download failed, fetching individually...")
                for symbol in valid_symbols:
                    results[symbol] = self.get_stock_data(symbol, period)
                    
        return results
    
    def validate_symbol(self, symbol):
        """
        Quick validation of stock symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info and ('symbol' in info or 'shortName' in info))
        except:
            return False
    
    def get_available_periods(self):
        """
        Get list of available periods for yfinance
        """
        return ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    def get_available_intervals(self):
        """
        Get list of available intervals for yfinance
        """
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

# Global instance
@st.cache_resource
def get_data_fetcher():
    """
    Get cached data fetcher instance
    """
    return StockDataFetcher()

def fetch_stock_data_safe(symbol, period='1y', interval='1d'):
    """
    Safe wrapper function for fetching stock data
    """
    fetcher = get_data_fetcher()
    return fetcher.get_stock_data(symbol, period, interval)

def fetch_company_info_safe(symbol):
    """
    Safe wrapper function for fetching company info
    """
    fetcher = get_data_fetcher()
    return fetcher.get_company_info(symbol)

def fetch_multiple_stocks_safe(symbols, period='1y'):
    """
    Safe wrapper function for fetching multiple stocks
    """
    fetcher = get_data_fetcher()
    return fetcher.get_multiple_stocks(symbols, period)