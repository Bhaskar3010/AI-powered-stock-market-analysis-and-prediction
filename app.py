"""
Module 7: Interactive Analytical Presentation Layer
Assembles downstream metrics into a robust dashboard interface using Streamlit and Plotly.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from ai_scorer import AIScorer
from news_fetcher import NewsFetcher

from utils import transform_dataframe_to_csv, format_large_numbers

# Configure layout constraints before rendering components
st.set_page_config(
    page_title="AI-Driven Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize application layers
fetcher = DataFetcher()
scorer = AIScorer()
scorer.load_model()
news_fetcher = NewsFetcher()


# Initialize session state for caching
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'pipeline_cache_time' not in st.session_state:
    st.session_state.pipeline_cache_time = None

@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API rate limits
def execute_pipeline_processing_loop(limit_tickers: int = 50, news_enabled: bool = True) -> pd.DataFrame:
    """
    Runs the data aggregation and algorithmic scoring loop across loaded configurations.
    Defaults to first 50 tickers for better performance. Increase limit_tickers to process more.
    """
    try:
        universe_df = fetcher.load_universe()
    except Exception as e:
        st.error(f"Failed to load stock list: {str(e)}")
        return pd.DataFrame()
    
    # Limit to first N tickers for performance
    universe_df = universe_df.head(limit_tickers)
    
    processed_records = []
    
    # Progress visualization indicator during execution
    progress_bar = st.progress(0)
    total_tickers = len(universe_df)
    
    for idx, row in universe_df.iterrows():
        ticker = row['Ticker']
        company_name = row['Company']
        sector = row['Sector']
        
        # Update progress bar safely
        progress_bar.progress(min((idx + 1) / total_tickers, 1.0))
        
        # Ingestion phase
        ohlcv_df = fetcher.fetch_historical_ohlcv(ticker, period="1y")
        if ohlcv_df.empty or len(ohlcv_df) < 20:
            continue
            
        fundamentals = fetcher.fetch_fundamental_metrics(ticker)
        
        # Analytical feature extraction phase
        ohlcv_df = IndicatorEngine.compute_technical_indicators(ohlcv_df)

        if news_enabled:
            news_features = news_fetcher.fetch_news_features(ticker, company_name=company_name)
        else:
            news_features = {}

        features = IndicatorEngine.extract_current_features(ohlcv_df, fundamentals, news_features)

        # Algorithm inference evaluation matrix execution
        ai_score = scorer.calculate_normalized_score(features)
        
        last_close = float(ohlcv_df['Close'].iloc[-1])
        last_open = float(ohlcv_df['Open'].iloc[-1])
        last_high = float(ohlcv_df['High'].iloc[-1])
        last_rsi = float(ohlcv_df['RSI'].iloc[-1])
        
        # Safely convert fundamental metrics to floats
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        pe_ratio = safe_float(fundamentals['pe_ratio'], 25.0)
        eps_growth = safe_float(fundamentals['eps_growth'], 0.05)
        market_cap = safe_float(fundamentals['market_cap'], 1e10)
        revenue_growth = safe_float(fundamentals['revenue_growth'], 0.05)
        
        processed_records.append({
            "Ticker": ticker,
            "Company Name": company_name,
            "Sector": sector,
            "AI Score": round(ai_score, 2),
            "Open Price (₹)": round(last_open, 2),
            "High Price (₹)": round(last_high, 2),
            "Close Price (₹)": round(last_close, 2),
            "RSI (14)": round(last_rsi, 2),
            "P/E Ratio": round(pe_ratio, 2),
            "Market Cap": market_cap,
            "EPS Growth (%)": round(eps_growth * 100, 2),
            "Revenue Growth (%)": round(revenue_growth * 100, 2),
            "News Sentiment Avg": round(float(features.get("News_Sentiment_Avg", 0.0)), 4),
            "News Count 24h": round(float(features.get("News_Count_24h", 0.0)), 0)
        })

        
    progress_bar.empty()
    
    if not processed_records:
        return pd.DataFrame()
        
    master_df = pd.DataFrame(processed_records)
    # Sort and return the top 50 ranked stocks based on their AI score
    master_df = master_df.sort_values(by="AI Score", ascending=False).head(50).reset_index(drop=True)
    return master_df

# UI Presentation Layout Definitions
st.title("📈 AI-Driven Stock Prediction System")
st.caption("Analyzing NSE/BSE Top Equities via Intelligence-Driven Ranking & Technical Indicators")

# Add settings to sidebar BEFORE running pipeline
st.sidebar.header("⚙️ Pipeline Settings")
ticker_limit = st.sidebar.slider(
    "Number of tickers to analyze (lower = faster)",
    min_value=10,
    max_value=500,
    value=50,
    step=10,
    help="Start with 50 for better performance. Increase to analyze more stocks (up to the 500-stock universe)."
)
force_refresh = st.sidebar.checkbox("Force refresh data (ignores cache)", value=False)
news_enabled = st.sidebar.toggle(
    "Enable News Analysis (slower, but richer features)",
    value=True
)

st.markdown("---")

# Data pipeline synchronization trigger
with st.spinner("Executing structural analysis metrics pipeline across universe..."):
    if force_refresh:
        # Clear cache if user wants fresh data
        execute_pipeline_processing_loop.clear()
    top_50_df = execute_pipeline_processing_loop(limit_tickers=ticker_limit, news_enabled=news_enabled)

if top_50_df.empty:
    st.error("System error: Unable to aggregate records or process tracking data arrays safely.")
else:
    # Sidebar Filtering Components
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Filter & Display")
    sectors = ["All"] + sorted(top_50_df['Sector'].unique().tolist())
    selected_sector = st.sidebar.selectbox("Filter Sector Classification Matrix", sectors)
    
    score_cutoff = st.sidebar.slider("Minimum AI Threshold Cutoff Score", 0.0, 100.0, 30.0)

    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Additional Filters (Optional)")

    additional_filters_enabled = st.sidebar.toggle(
        "Enable Additional Filters",
        value=True
    )

    if additional_filters_enabled:
        # Market cap buckets (simple ranges)
        st.sidebar.subheader("Market Cap Filters")
        market_cap_category = st.sidebar.selectbox(
            "Market Cap Category",
            options=["Any", "Large", "Mid", "Small"],
            index=0
        )
    else:
        market_cap_category = "Any"


    # Keep only market cap category filtering (Large/Mid/Small) without extra numeric thresholds.
    # Market Cap Category uses built-in thresholds in the filter logic below.
    # (₹ Cr sidebar numeric inputs were removed as per request.)
    market_cap_threshold_large = 1e11
    market_cap_threshold_mid = 5e10
    market_cap_threshold_small = 5e10


    pe_min = st.sidebar.number_input(
        "P/E min",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    pe_max = st.sidebar.number_input(
        "P/E max",
        min_value=0.0,
        value=float(1000.0),
        step=1.0
    )

    eps_growth_min = st.sidebar.number_input(
        "EPS Growth min (%)",
        min_value=-100.0,
        value=-100.0,
        step=0.5
    )

    eps_growth_max = st.sidebar.number_input(
        "EPS Growth max (%)",
        min_value=-100.0,
        value=1000.0,
        step=0.5
    )

    revenue_growth_min = st.sidebar.number_input(
        "Revenue Growth min (%)",
        min_value=-100.0,
        value=-100.0,
        step=0.5
    )

    revenue_growth_max = st.sidebar.number_input(
        "Revenue Growth max (%)",
        min_value=-100.0,
        value=1000.0,
        step=0.5
    )
    
    
    # Process analytical mutations based on user filter selections
    display_df = top_50_df.copy()
    if selected_sector != "All":
        display_df = display_df[display_df['Sector'] == selected_sector]
    display_df = display_df[display_df['AI Score'] >= score_cutoff]

    # Apply optional fundamental filters
    # Market cap category filter
    # Inputs are in ₹ Cr; DataFrame values are in absolute ₹.
    market_cap_threshold_large_abs = market_cap_threshold_large * 1e7
    market_cap_threshold_mid_abs = market_cap_threshold_mid * 1e7
    market_cap_threshold_small_abs = market_cap_threshold_small * 1e7

    if market_cap_category == "Large":
        display_df = display_df[display_df['Market Cap'] >= market_cap_threshold_large_abs]
    elif market_cap_category == "Mid":
        display_df = display_df[
            (display_df['Market Cap'] >= market_cap_threshold_mid_abs) & (display_df['Market Cap'] < market_cap_threshold_large_abs)
        ]
    elif market_cap_category == "Small":
        display_df = display_df[display_df['Market Cap'] <= market_cap_threshold_small_abs]


    # P/E filter (range)
    display_df = display_df[(display_df['P/E Ratio'] >= pe_min) & (display_df['P/E Ratio'] <= pe_max)]

    # EPS Growth (%) filter
    display_df = display_df[(display_df['EPS Growth (%)'] >= eps_growth_min) & (display_df['EPS Growth (%)'] <= eps_growth_max)]

    # Revenue Growth (%) filter
    display_df = display_df[(display_df['Revenue Growth (%)'] >= revenue_growth_min) & (display_df['Revenue Growth (%)'] <= revenue_growth_max)]
    
    # Render High-Level Statistics Metrics Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Scanned Universe Entities Available", len(display_df))
    with m2:
        st.metric("Top Matrix Optimization Score", f"{top_50_df['AI Score'].max() if not top_50_df.empty else 0.0} / 100")
    with m3:
        st.metric("Data Refresh Sync Context Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M'))
        
    # Main Screener Results Table
    st.subheader("Top 50 Al-Ranked Investment Engine Matrix")
    
    formatted_display_df = display_df.copy()
    formatted_display_df['Market Cap'] = formatted_display_df['Market Cap'].apply(format_large_numbers)
    
    st.dataframe(
        formatted_display_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol"),
            "AI Score": st.column_config.ProgressColumn("AI Score Engine Strength", min_value=0, max_value=100, format="%.2f"),
            "Open Price (₹)": st.column_config.NumberColumn("Last Open Price", format="₹%.2f"),
            "High Price (₹)": st.column_config.NumberColumn("Last High Price", format="₹%.2f"),
            "Close Price (₹)": st.column_config.NumberColumn("Last Close Price", format="₹%.2f")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # CSV Export Controls Trigger Area
    csv_bytes = transform_dataframe_to_csv(display_df)
    st.download_button(
        label="📥 Export Screened Selection Matrix to CSV Format",
        data=csv_bytes,
        file_name = f"AIDSP_Top_Stocks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Detailed Stock Analytics & Interactive Plotly Charts Section
    st.subheader("Detailed Stock Performance Workspace")
    target_ticker = st.selectbox("Select Target Ticker Asset for Interactive Core Deep-Dive Analysis", display_df['Ticker'].unique())
    
    if target_ticker:
        # Fetch detailed OHLCV with a resilient fallback strategy.
        # Some tickers occasionally return empty data for narrower periods (e.g., "6m").
        chart_ohlcv_6m = fetcher.fetch_historical_ohlcv(target_ticker, period="6m")
        used_period = "6m"

        if chart_ohlcv_6m.empty:
            chart_ohlcv = fetcher.fetch_historical_ohlcv(target_ticker, period="1y")
            used_period = "1y"
        else:
            chart_ohlcv = chart_ohlcv_6m

        if not chart_ohlcv.empty:
            chart_ohlcv = IndicatorEngine.compute_technical_indicators(chart_ohlcv)
            
            # Construct interactive charts using Plotly Subplots configuration logic
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=chart_ohlcv['Date'],
                open=chart_ohlcv['Open'],
                high=chart_ohlcv['High'],
                low=chart_ohlcv['Low'],
                close=chart_ohlcv['Close'],
                name="Price Actions Bar"
            ))
            
            # Overlay Moving Averages
            fig.add_trace(go.Scatter(x=chart_ohlcv['Date'], y=chart_ohlcv['SMA_50'], line=dict(color='orange', width=1.5), name="SMA 50 Timeline"))
            fig.add_trace(go.Scatter(x=chart_ohlcv['Date'], y=chart_ohlcv['SMA_200'], line=dict(color='red', width=1.5), name="SMA 200 Timeline"))
            
            fig.update_layout(
                title=f"{target_ticker} - {used_period} Technical Pricing Analysis Dashboard Canvas",
                yaxis_title="Asset Price (₹)",
                xaxis_title="Timeline Interval Periods",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Separate volume chart
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(
                x=chart_ohlcv['Date'],
                y=chart_ohlcv['Volume'],
                name="Trading Volume Volume Matrix Bars",
                marker_color='lightblue'
            ))
            vol_fig.update_layout(
                title=f"{target_ticker} Market Execution Volume Density Trends ({used_period})",
                yaxis_title="Volume Metrics Delta",
                xaxis_title="Timeline Interval Periods",
                height=250,
                template="plotly_dark"
            )
            st.plotly_chart(vol_fig, use_container_width=True)
        else:
            st.warning(
                "Unable to gather structural historical metrics details for selected target ticker asset. "
                f"(Attempted periods: 6m and 1y; both returned empty for: {target_ticker})"
            )

# Footer compliance disclaimer display section
st.markdown("---")
st.info("⚠️ **Disclaimer:** This tool is for informational purposes only and does not constitute financial advice.")
