import streamlit as st
import os
import google.generativeai as genai
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# Configure Gemini AI
def initialize_genai():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY environment variable")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {str(e)}")
        return False

def get_gemini_response(question, model, chat):
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error getting response from Gemini: {str(e)}")
        return None

def get_stock_data(symbol, timeframe, period='1y'):
    """Fetch stock data based on timeframe"""
    intervals = {
        'Intraday': '15m',
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo',
        'Yearly': '1y'
    }
    
    try:
        data = yf.download(symbol, period=period, interval=intervals[timeframe])
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    if df is None or df.empty:
        return None
    
    # Calculate Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # Bollinger Bands
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    return df

def plot_technical_analysis(df, selected_indicators):
    """Create technical analysis plots"""
    if df is None or df.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2])

    # Add candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'),
                  row=1, col=1)

    # Add selected indicators
    if 'Moving Averages' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                               line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                               line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
                               line=dict(color='purple', width=1)), row=1, col=1)

    if 'Bollinger Bands' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                               line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                               line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='BB Middle',
                               line=dict(color='gray', width=1)), row=1, col=1)

    if 'RSI' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                               line=dict(color='green', width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)

    if 'MACD' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                               line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal',
                               line=dict(color='orange', width=1)), row=3, col=1)

    # Update layout
    fig.update_layout(
        title_text="Technical Analysis Chart",
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def main():
    st.set_page_config(
        page_title="Advanced Stock Market Prediction App",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Advanced Stock Market Prediction App")
    st.markdown("---")

    # Initialize Gemini AI
    if not initialize_genai():
        return

    # Initialize model and chat
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    chat = model.start_chat(history=[])

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create input columns
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_input = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", 
                                   key="input",
                                   placeholder="Enter stock symbol...")
    
    with col2:
        timeframe = st.selectbox("Select Timeframe:", 
                               ["Intraday", "Daily", "Weekly", "Monthly", "Yearly"])
    
    with col3:
        indicators = st.multiselect("Select Technical Indicators:",
                                  ["Moving Averages", "Bollinger Bands", "RSI", "MACD"],
                                  default=["Moving Averages"])

    analyze = st.button("Analyze Stock", type="primary")

    if analyze and stock_input:
        with st.spinner(f"Analyzing {stock_input.upper()} stock..."):
            # Fetch and process stock data
            df = get_stock_data(stock_input.upper(), timeframe)
            
            if df is not None:
                # Calculate technical indicators
                df_with_indicators = calculate_technical_indicators(df)
                
                # Plot technical analysis
                fig = plot_technical_analysis(df_with_indicators, indicators)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Get AI prediction
                prompt = f"""
                Analyze {stock_input.upper()} stock based on the following timeframe: {timeframe}
                
                Please provide:
                1. Technical Analysis Summary:
                   - Price trend analysis
                   - Support and resistance levels
                   - Volume analysis
                   
                2. Selected Indicators Analysis:
                   {', '.join(indicators)}
                   
                3. Predicted price ranges for:
                   - Short-term (1-3 months)
                   - Medium-term (3-6 months)
                   - Long-term (6-12 months)
                
                4. Key risks and opportunities
                
                Format the analysis in a clear, structured manner.
                """
                
                response = get_gemini_response(prompt, model, chat)
                
                if response:
                    st.markdown("### 🎯 Analysis Results")
                    
                    with st.container():
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        
                        # Display streaming response
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in response:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stock": stock_input.upper(),
                        "timeframe": timeframe,
                        "indicators": indicators,
                        "response": full_response
                    })

    # Display chat history
    if st.session_state.chat_history:
        with st.expander("📜 Previous Analyses"):
            for entry in reversed(st.session_state.chat_history):
                st.markdown(f"**{entry['stock']} - {entry['timeframe']} - {entry['timestamp']}**")
                st.markdown(f"*Indicators analyzed: {', '.join(entry['indicators'])}*")
                st.markdown(entry['response'])
                st.markdown("---")

    # Add disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This app provides AI-generated predictions and technical analysis for educational purposes only. 
    These predictions should not be considered as financial advice. Always conduct your own research 
    and consult with qualified financial advisors before making investment decisions.
    """)

if __name__ == "__main__":
    main()
    
