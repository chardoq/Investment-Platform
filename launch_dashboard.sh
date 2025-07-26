#!/bin/bash

echo "🚀 Starting Investment Analysis Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the dashboard"
echo ""

# Activate virtual environment and run dashboard
source investment_env/bin/activate
streamlit run investment_dashboard.py --server.headless false --server.port 8501