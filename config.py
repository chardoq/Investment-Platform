# Configuration file for Investment Dashboard

# Default stock symbols for quick access
DEFAULT_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS',
    'ADBE', 'NFLX', 'CRM', 'XOM', 'KO', 'PFE', 'INTC', 'CSCO', 'VZ', 'T'
]

# Popular ETFs for portfolio diversification
POPULAR_ETFS = [
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq 100 ETF
    'IWM',   # Russell 2000 ETF
    'EFA',   # MSCI EAFE ETF
    'EEM',   # Emerging Markets ETF
    'TLT',   # 20+ Year Treasury Bond ETF
    'GLD',   # Gold ETF
    'VTI',   # Total Stock Market ETF
    'BND',   # Total Bond Market ETF
    'ARKK',  # ARK Innovation ETF
]

# Market sectors for analysis
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
}

# Risk-free rate (10-year Treasury yield approximation)
RISK_FREE_RATE = 0.045

# Default date ranges
DATE_RANGES = {
    '1 Month': 30,
    '3 Months': 90,
    '6 Months': 180,
    '1 Year': 365,
    '2 Years': 730,
    '5 Years': 1825
}

# Chart colors
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'background': '#ffffff'
}