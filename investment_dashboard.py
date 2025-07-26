import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import ta
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Import our robust data fetcher
from data_fetcher import fetch_stock_data_safe, fetch_company_info_safe, fetch_multiple_stocks_safe

# Page configuration
st.set_page_config(
    page_title="Investment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e6e9ef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class InvestmentAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def get_stock_data(self, symbols, period="1y"):
        """Fetch stock data for given symbols using robust data fetcher"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if len(symbols) == 1:
            data = fetch_stock_data_safe(symbols[0], period)
            return {symbols[0]: data} if not data.empty else {}
        else:
            return fetch_multiple_stocks_safe(symbols, period)
    
    def get_company_info(self, symbol):
        """Get company information using robust data fetcher"""
        return fetch_company_info_safe(symbol)
    
    def calculate_returns(self, data):
        """Calculate various return metrics"""
        if data.empty:
            return {}
        
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        
        # Price change metrics
        total_return = (current_price - start_price) / start_price
        
        # Daily returns
        daily_returns = data['Close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = daily_returns.mean() * 252 - self.risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': daily_returns.mean() * 252,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': daily_returns
        }
    
    def technical_analysis(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb_indicator.bollinger_hband()
        df['BB_lower'] = bb_indicator.bollinger_lband()
        df['BB_middle'] = bb_indicator.bollinger_mavg()
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
        
        return df
    
    def portfolio_optimization(self, symbols, period="1y"):
        """Modern Portfolio Theory optimization"""
        try:
            # Get data for all symbols
            data = {}
            returns_data = {}
            
            for symbol in symbols:
                ticker_data = self.get_stock_data(symbol, period)
                if symbol in ticker_data and not ticker_data[symbol].empty:
                    data[symbol] = ticker_data[symbol]
                    returns_data[symbol] = ticker_data[symbol]['Close'].pct_change().dropna()
            
            if len(returns_data) < 2:
                return None, None
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # Calculate metrics
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # Number of assets
            num_assets = len(symbols)
            
            # Optimization function
            def portfolio_stats(weights, mean_returns, cov_matrix):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
                return portfolio_return, portfolio_std, sharpe_ratio
            
            def negative_sharpe(weights, mean_returns, cov_matrix):
                return -portfolio_stats(weights, mean_returns, cov_matrix)[2]
            
            # Constraints and bounds
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = num_assets * [1. / num_assets]
            
            # Optimize for maximum Sharpe ratio
            result = minimize(negative_sharpe, initial_guess, 
                            args=(mean_returns, cov_matrix),
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                opt_return, opt_volatility, opt_sharpe = portfolio_stats(
                    optimal_weights, mean_returns, cov_matrix
                )
                
                portfolio_metrics = {
                    'Expected Return': opt_return,
                    'Volatility': opt_volatility,
                    'Sharpe Ratio': opt_sharpe,
                    'Weights': dict(zip(symbols, optimal_weights))
                }
                
                return portfolio_metrics, returns_df
            else:
                return None, None
                
        except Exception as e:
            st.error(f"Portfolio optimization error: {str(e)}")
            return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Investment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = InvestmentAnalyzer()
    
    # Sidebar for navigation and inputs
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Investment+Pro", width=200)
        
        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=["Stock Analysis", "Portfolio Optimizer", "Market Overview", "Technical Analysis"],
            icons=["graph-up", "pie-chart", "globe", "activity"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
        
        st.markdown("---")
        
        # Common inputs
        if selected in ["Stock Analysis", "Technical Analysis"]:
            default_symbol = st.text_input("Stock Symbol", "AAPL", help="Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)")
            period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        elif selected == "Portfolio Optimizer":
            st.subheader("Portfolio Stocks")
            num_stocks = st.number_input("Number of stocks", min_value=2, max_value=10, value=4)
            symbols = []
            for i in range(num_stocks):
                symbol = st.text_input(f"Stock {i+1}", value=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "JNJ", "V", "WMT"][i] if i < 10 else "")
                if symbol:
                    symbols.append(symbol.upper())
            period = st.selectbox("Analysis Period", ["6mo", "1y", "2y", "3y"], index=1)
    
    # Main content area
    if selected == "Stock Analysis":
        st.subheader(f"📊 Analysis for {default_symbol}")
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            stock_data = analyzer.get_stock_data(default_symbol, period)
            company_info = analyzer.get_company_info(default_symbol)
        
        if default_symbol in stock_data:
            data = stock_data[default_symbol]
            
            # Company info section
            if company_info:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Market Cap</h3>
                        <h2>${company_info.get('marketCap', 0):,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>P/E Ratio</h3>
                        <h2>{company_info.get('trailingPE', 'N/A')}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>52W High</h3>
                        <h2>${company_info.get('fiftyTwoWeekHigh', 0):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>52W Low</h3>
                        <h2>${company_info.get('fiftyTwoWeekLow', 0):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{default_symbol} Stock Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            returns_metrics = analyzer.calculate_returns(data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    'Value': [
                        f"{returns_metrics['total_return']:.2%}",
                        f"{returns_metrics['annualized_return']:.2%}",
                        f"{returns_metrics['volatility']:.2%}",
                        f"{returns_metrics['sharpe_ratio']:.2f}",
                        f"{returns_metrics['max_drawdown']:.2%}"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("📊 Returns Distribution")
                returns = returns_metrics['daily_returns']
                fig_hist = px.histogram(
                    x=returns,
                    nbins=50,
                    title="Daily Returns Distribution",
                    labels={'x': 'Daily Returns', 'y': 'Frequency'}
                )
                fig_hist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Volume analysis
            st.subheader("📊 Volume Analysis")
            fig_volume = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            fig_volume.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig_volume.add_trace(
                go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='orange'),
                row=2, col=1
            )
            
            fig_volume.update_layout(
                template="plotly_white",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        else:
            st.error(f"Unable to fetch data for {default_symbol}. Please check the symbol and try again.")
    
    elif selected == "Portfolio Optimizer":
        st.subheader("🎯 Portfolio Optimization")
        
        if len(symbols) >= 2:
            with st.spinner("Optimizing portfolio..."):
                portfolio_metrics, returns_df = analyzer.portfolio_optimization(symbols, period)
            
            if portfolio_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Optimal Portfolio Metrics")
                    metrics_display = pd.DataFrame({
                        'Metric': ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                        'Value': [
                            f"{portfolio_metrics['Expected Return']:.2%}",
                            f"{portfolio_metrics['Volatility']:.2%}",
                            f"{portfolio_metrics['Sharpe Ratio']:.3f}"
                        ]
                    })
                    st.dataframe(metrics_display, hide_index=True, use_container_width=True)
                
                with col2:
                    st.subheader("🥧 Optimal Allocation")
                    weights_df = pd.DataFrame(
                        list(portfolio_metrics['Weights'].items()),
                        columns=['Symbol', 'Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(weights_df, hide_index=True, use_container_width=True)
                
                # Pie chart for allocation
                fig_pie = px.pie(
                    values=list(portfolio_metrics['Weights'].values()),
                    names=list(portfolio_metrics['Weights'].keys()),
                    title="Optimal Portfolio Allocation"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(template="plotly_white", height=500)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Correlation matrix
                st.subheader("🔄 Correlation Matrix")
                corr_matrix = returns_df.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Asset Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig_corr.update_layout(template="plotly_white", height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Risk-Return scatter plot
                st.subheader("📈 Risk vs Return Analysis")
                individual_metrics = []
                for symbol in symbols:
                    symbol_data = analyzer.get_stock_data(symbol, period)
                    if symbol in symbol_data:
                        returns = analyzer.calculate_returns(symbol_data[symbol])
                        individual_metrics.append({
                            'Symbol': symbol,
                            'Return': returns['annualized_return'],
                            'Risk': returns['volatility'],
                            'Sharpe': returns['sharpe_ratio']
                        })
                
                if individual_metrics:
                    metrics_df = pd.DataFrame(individual_metrics)
                    
                    fig_scatter = px.scatter(
                        metrics_df,
                        x='Risk',
                        y='Return',
                        text='Symbol',
                        color='Sharpe',
                        size='Sharpe',
                        title="Risk vs Return Analysis",
                        labels={'Risk': 'Annual Volatility', 'Return': 'Annual Return'}
                    )
                    
                    # Add portfolio point
                    fig_scatter.add_trace(go.Scatter(
                        x=[portfolio_metrics['Volatility']],
                        y=[portfolio_metrics['Expected Return']],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name='Optimal Portfolio',
                        text=['Portfolio'],
                        textposition='top center'
                    ))
                    
                    fig_scatter.update_traces(textposition='top center')
                    fig_scatter.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            else:
                st.error("Unable to optimize portfolio. Please check the symbols and try again.")
        
        else:
            st.warning("Please enter at least 2 stock symbols for portfolio optimization.")
    
    elif selected == "Technical Analysis":
        st.subheader(f"🔧 Technical Analysis for {default_symbol}")
        
        with st.spinner("Calculating technical indicators..."):
            stock_data = analyzer.get_stock_data(default_symbol, period)
        
        if default_symbol in stock_data:
            data = stock_data[default_symbol]
            technical_data = analyzer.technical_analysis(data)
            
            # Technical indicators chart
            fig_tech = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Price & Moving Averages', 'MACD', 'RSI', 'Bollinger Bands'),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price and moving averages
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
            
            # MACD
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD_signal'], name='Signal', line=dict(color='red')), row=2, col=1)
            
            # RSI
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Bollinger Bands
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Close'], name='Close', line=dict(color='blue')), row=4, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_upper'], name='BB Upper', line=dict(color='red', dash='dash')), row=4, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_lower'], name='BB Lower', line=dict(color='green', dash='dash')), row=4, col=1)
            fig_tech.add_trace(go.Scatter(x=technical_data.index, y=technical_data['BB_middle'], name='BB Middle', line=dict(color='orange')), row=4, col=1)
            
            fig_tech.update_layout(
                template="plotly_white",
                height=1000,
                showlegend=False,
                title=f"Technical Analysis for {default_symbol}"
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # Current indicator values
            st.subheader("📊 Current Indicator Values")
            current_indicators = {
                'RSI': technical_data['RSI'].iloc[-1],
                'MACD': technical_data['MACD'].iloc[-1],
                'MACD Signal': technical_data['MACD_signal'].iloc[-1],
                'SMA 20': technical_data['SMA_20'].iloc[-1],
                'SMA 50': technical_data['SMA_50'].iloc[-1],
                'BB Upper': technical_data['BB_upper'].iloc[-1],
                'BB Lower': technical_data['BB_lower'].iloc[-1]
            }
            
            indicators_df = pd.DataFrame(
                list(current_indicators.items()),
                columns=['Indicator', 'Value']
            )
            indicators_df['Value'] = indicators_df['Value'].apply(lambda x: f"{x:.2f}")
            st.dataframe(indicators_df, hide_index=True, use_container_width=True)
            
            # Trading signals
            st.subheader("🎯 Trading Signals")
            signals = []
            
            current_rsi = technical_data['RSI'].iloc[-1]
            if current_rsi > 70:
                signals.append("🔴 RSI indicates OVERBOUGHT condition")
            elif current_rsi < 30:
                signals.append("🟢 RSI indicates OVERSOLD condition")
            else:
                signals.append("🟡 RSI in neutral zone")
            
            current_price = technical_data['Close'].iloc[-1]
            sma_20 = technical_data['SMA_20'].iloc[-1]
            sma_50 = technical_data['SMA_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                signals.append("🟢 Price above both moving averages - BULLISH")
            elif current_price < sma_20 < sma_50:
                signals.append("🔴 Price below both moving averages - BEARISH")
            else:
                signals.append("🟡 Mixed moving average signals")
            
            macd = technical_data['MACD'].iloc[-1]
            macd_signal = technical_data['MACD_signal'].iloc[-1]
            
            if macd > macd_signal:
                signals.append("🟢 MACD above signal line - BULLISH")
            else:
                signals.append("🔴 MACD below signal line - BEARISH")
            
            for signal in signals:
                st.write(signal)
        
        else:
            st.error(f"Unable to fetch data for {default_symbol}. Please check the symbol and try again.")
    
    elif selected == "Market Overview":
        st.subheader("🌍 Market Overview")
        
        # Major indices
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        with st.spinner("Fetching market data..."):
            market_data = {}
            for name, symbol in indices.items():
                data = analyzer.get_stock_data(symbol, "5d")
                if symbol in data:
                    market_data[name] = data[symbol]
        
        if market_data:
            # Market overview cards
            cols = st.columns(len(market_data))
            
            for i, (name, data) in enumerate(market_data.items()):
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                color = "🟢" if change >= 0 else "🔴"
                
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{name}</h4>
                        <h3>{current_price:.2f}</h3>
                        <p>{color} {change:+.2f} ({change_pct:+.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Market performance chart
            st.subheader("📈 Market Performance (5 Days)")
            
            fig_market = go.Figure()
            
            for name, data in market_data.items():
                if name != "VIX":  # Exclude VIX from price chart
                    normalized_prices = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                    fig_market.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized_prices,
                        mode='lines',
                        name=name,
                        hovertemplate=f'{name}<br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'
                    ))
            
            fig_market.update_layout(
                title="Market Indices Performance (% Change)",
                xaxis_title="Date",
                yaxis_title="Percentage Change (%)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_market, use_container_width=True)
            
            # Sector performance (sample data - in real implementation, you'd fetch sector ETFs)
            st.subheader("🏭 Sector Performance")
            
            sector_symbols = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial": "XLF",
                "Energy": "XLE",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Industrials": "XLI",
                "Real Estate": "XLRE"
            }
            
            sector_performance = []
            
            with st.spinner("Fetching sector data..."):
                for sector, symbol in sector_symbols.items():
                    data = analyzer.get_stock_data(symbol, "1mo")
                    if symbol in data and not data[symbol].empty:
                        current = data[symbol]['Close'].iloc[-1]
                        start = data[symbol]['Close'].iloc[0]
                        performance = ((current - start) / start) * 100
                        sector_performance.append({
                            'Sector': sector,
                            'Performance': performance
                        })
            
            if sector_performance:
                sector_df = pd.DataFrame(sector_performance)
                sector_df = sector_df.sort_values('Performance', ascending=True)
                
                fig_sectors = px.bar(
                    sector_df,
                    x='Performance',
                    y='Sector',
                    orientation='h',
                    title="Sector Performance (1 Month)",
                    color='Performance',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                
                fig_sectors.update_layout(
                    template="plotly_white",
                    height=500,
                    xaxis_title="Performance (%)"
                )
                
                st.plotly_chart(fig_sectors, use_container_width=True)
        
        else:
            st.error("Unable to fetch market data. Please try again later.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        <p>📈 Investment Analysis Dashboard | Built with Streamlit & Financial APIs</p>
        <p><em>Disclaimer: This dashboard is for educational purposes only. Not financial advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
