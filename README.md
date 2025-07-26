# 📈 Investment Analysis Dashboard

A comprehensive Streamlit-based investment analysis dashboard that helps analyze stocks and craft investment portfolios using modern financial metrics and technical analysis.

## 🚀 Features

### 📊 Stock Analysis
- **Company Information**: Market cap, P/E ratio, 52-week high/low
- **Performance Metrics**: Total return, annualized return, volatility, Sharpe ratio, maximum drawdown
- **Interactive Price Charts**: Real-time stock price visualization with hover details
- **Returns Distribution**: Histogram analysis of daily returns
- **Volume Analysis**: Combined price and volume charts

### 🎯 Portfolio Optimization
- **Modern Portfolio Theory**: Optimal portfolio allocation using mean-variance optimization
- **Risk-Return Analysis**: Scatter plot showing individual stocks vs optimal portfolio
- **Correlation Matrix**: Asset correlation heatmap
- **Portfolio Metrics**: Expected return, volatility, and Sharpe ratio
- **Visual Allocation**: Interactive pie charts showing optimal weights

### 🔧 Technical Analysis
- **Moving Averages**: SMA 20, SMA 50, EMA 12, EMA 26
- **MACD**: Moving Average Convergence Divergence with signal line
- **RSI**: Relative Strength Index with overbought/oversold levels
- **Bollinger Bands**: Price volatility bands
- **Trading Signals**: Automated buy/sell signal generation

### 🌍 Market Overview
- **Market Indices**: S&P 500, Dow Jones, NASDAQ, Russell 2000, VIX
- **Sector Performance**: Real-time sector ETF performance tracking
- **Market Performance Charts**: Normalized performance comparison

## 🛠️ Installation

1. **Clone the repository or create a new directory:**
   ```bash
   mkdir investment-dashboard
   cd investment-dashboard
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run investment_dashboard.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

## 📦 Dependencies

- **streamlit**: Web application framework
- **yfinance**: Yahoo Finance API for stock data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing for optimization
- **ta**: Technical analysis library
- **streamlit-option-menu**: Enhanced navigation menu

## 🎯 Usage Guide

### Stock Analysis
1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) in the sidebar
2. Select your desired time period
3. View comprehensive analysis including:
   - Company fundamentals
   - Price performance charts
   - Risk metrics
   - Volume analysis

### Portfolio Optimization
1. Navigate to "Portfolio Optimizer"
2. Enter 2-10 stock symbols
3. Select analysis period
4. View optimized portfolio allocation based on:
   - Maximum Sharpe ratio
   - Risk-return optimization
   - Correlation analysis

### Technical Analysis
1. Select "Technical Analysis"
2. Enter stock symbol and time period
3. Analyze multiple technical indicators
4. Review automated trading signals

### Market Overview
1. View real-time market indices performance
2. Compare sector performance
3. Monitor market trends and volatility

## 📊 Key Metrics Explained

- **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **RSI**: Momentum oscillator (70+ overbought, 30- oversold)
- **MACD**: Trend-following momentum indicator

## 🔮 Advanced Features

### Portfolio Optimization Algorithm
The dashboard uses Modern Portfolio Theory (MPT) with:
- Mean-variance optimization
- Constraints: weights sum to 1, no short selling
- Objective: Maximize Sharpe ratio
- Risk-free rate: 2% (configurable)

### Technical Indicators
- **Moving Averages**: Trend identification
- **Bollinger Bands**: Volatility and price level analysis
- **RSI**: Momentum and reversal signals
- **MACD**: Trend changes and momentum shifts

### Data Sources
- **Yahoo Finance**: Real-time stock prices and company data
- **Market Indices**: Major US market benchmarks
- **Sector ETFs**: Sector performance tracking

## ⚠️ Disclaimer

This dashboard is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or professional guidance. Always consult with qualified financial advisors before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New technical indicators
- Additional data sources
- UI/UX improvements
- Performance optimizations

## 📝 License

This project is open source and available under the MIT License.

## 🎨 Customization

The dashboard features a modern, responsive design with:
- Custom CSS styling
- Interactive charts
- Professional color schemes
- Mobile-friendly layout

You can customize the appearance by modifying the CSS in the `st.markdown()` sections of the code.

## 🔧 Configuration

Key parameters that can be adjusted:
- Risk-free rate (currently 2%)
- Technical indicator periods
- Chart colors and styling
- Default stock symbols

## 📈 Future Enhancements

Planned features:
- Real-time alerts and notifications
- More advanced portfolio strategies
- Options pricing models
- Cryptocurrency analysis
- Machine learning predictions
- Export functionality for reports