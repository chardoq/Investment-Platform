#!/usr/bin/env python3
"""
Test script to verify the investment dashboard data fetching functionality
"""

import sys
import os
sys.path.append('.')

from data_fetcher import fetch_stock_data_safe, fetch_company_info_safe, fetch_multiple_stocks_safe

def test_single_stock():
    """Test fetching data for a single stock"""
    print("🔍 Testing single stock data fetching...")
    
    symbol = 'AAPL'
    data = fetch_stock_data_safe(symbol, '1mo')
    
    if not data.empty:
        print(f"✅ Successfully fetched data for {symbol}")
        print(f"   Data shape: {data.shape}")
        print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        return True
    else:
        print(f"❌ Failed to fetch data for {symbol}")
        return False

def test_company_info():
    """Test fetching company information"""
    print("\n🏢 Testing company information fetching...")
    
    symbol = 'AAPL'
    info = fetch_company_info_safe(symbol)
    
    if info:
        print(f"✅ Successfully fetched company info for {symbol}")
        print(f"   Company name: {info.get('longName', 'N/A')}")
        print(f"   Sector: {info.get('sector', 'N/A')}")
        print(f"   Market cap: ${info.get('marketCap', 0):,}")
        print(f"   P/E ratio: {info.get('trailingPE', 'N/A')}")
        return True
    else:
        print(f"❌ Failed to fetch company info for {symbol}")
        return False

def test_multiple_stocks():
    """Test fetching data for multiple stocks"""
    print("\n📊 Testing multiple stocks data fetching...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = fetch_multiple_stocks_safe(symbols, '5d')
    
    if data:
        print(f"✅ Successfully fetched data for {len(data)} stocks")
        for symbol, stock_data in data.items():
            if not stock_data.empty:
                print(f"   {symbol}: {len(stock_data)} records, latest close: ${stock_data['Close'].iloc[-1]:.2f}")
            else:
                print(f"   {symbol}: No data")
        return True
    else:
        print("❌ Failed to fetch data for multiple stocks")
        return False

def test_invalid_symbol():
    """Test handling of invalid symbols"""
    print("\n⚠️  Testing invalid symbol handling...")
    
    symbol = 'INVALIDXYZ'
    data = fetch_stock_data_safe(symbol, '1mo')
    
    if data.empty:
        print(f"✅ Correctly handled invalid symbol {symbol}")
        return True
    else:
        print(f"❌ Should have returned empty data for invalid symbol {symbol}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Investment Dashboard Data Fetching")
    print("=" * 50)
    
    tests = [
        test_single_stock,
        test_company_info,
        test_multiple_stocks,
        test_invalid_symbol
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📈 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n💡 To run the dashboard:")
        print("   source investment_env/bin/activate")
        print("   streamlit run investment_dashboard.py")
        print("   Open http://localhost:8501 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)