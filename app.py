import streamlit as st
import pandas as pd
import altair as alt
from utils import detect_anomalies, get_anomaly_periods, get_explanation

# Check if PyTorch is available and set a flag
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Time-Series Anomaly Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Streamlit UI ---

st.title("🤖 Time-Series Anomaly Detector")
st.markdown("Upload your data, detect anomalies, and get AI-powered explanations for why they might have occurred.")

# Initialize session state for chat
if 'explaining_anomaly' not in st.session_state:
    st.session_state.explaining_anomaly = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not PYTORCH_AVAILABLE:
        st.warning("PyTorch not found. The Hybrid LSTM model will be disabled. Please run `pip install torch`.")

    uploaded_file = st.file_uploader("Upload your time-series CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success("File uploaded successfully!")
        
        st.subheader("Column Selection")
        date_col = st.selectbox("Select Date/Time Column", df.columns, index=0)
        value_col = st.selectbox("Select Value Column (Target)", df.columns, index=1 if len(df.columns) > 1 else 0)
        
        st.subheader("Model & Parameters")
        model_options = {
            'arima': "ARIMA (Forecasting Error)"
        }
        if PYTORCH_AVAILABLE:
            model_options['hybrid_lstm'] = "Hybrid LSTM (Reconstruction + Benford's Law)"
        
        model_choice = st.selectbox(
            "Detection Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        window_size = st.number_input("Window Size / Look-back Period", min_value=10, max_value=200, value=30, step=5)
        threshold = st.number_input("Anomaly Threshold (Std Dev)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

        if st.button("🚀 Detect Anomalies", use_container_width=True):
            st.session_state.date_col = date_col
            st.session_state.value_col = value_col
            st.session_state.model_choice = model_choice
            st.session_state.window_size = window_size
            st.session_state.threshold = threshold
            st.session_state.run_analysis = True
            # Reset chat state on new analysis
            st.session_state.explaining_anomaly = None
            st.session_state.explanation = None

# --- Main Content Area ---
if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    with st.spinner("Analyzing data... This may take a moment, especially for the Hybrid LSTM model."):
        df = st.session_state.df.copy()
        date_col = st.session_state.date_col
        value_col = st.session_state.value_col
        
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df[value_col] = pd.to_numeric(df[value_col])
            df = df.sort_values(by=date_col).reset_index(drop=True)
        except Exception as e:
            st.error(f"Error processing columns. Please ensure '{date_col}' is a valid date format and '{value_col}' is numeric. Details: {e}")
            st.stop()
            
        results_df = detect_anomalies(
            df, 
            value_col, 
            st.session_state.window_size, 
            st.session_state.threshold, 
            st.session_state.model_choice
        )
        
        if results_df is not None:
            anomaly_periods = get_anomaly_periods(results_df, date_col)
            st.session_state.results_df = results_df
            st.session_state.anomaly_periods = anomaly_periods
            st.session_state.display_results = True
        else:
            st.session_state.display_results = False


if st.session_state.get('display_results', False):
    st.header("📊 Analysis Results")
    results_df = st.session_state.results_df
    anomaly_periods = st.session_state.anomaly_periods
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col

    base_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X(f'{date_col}:T', title='Date'),
        y=alt.Y(f'{value_col}:Q', title='Value'),
        tooltip=[date_col, value_col]
    ).interactive()

    anomaly_regions = alt.Chart(anomaly_periods).mark_rect(opacity=0.3, color='red').encode(
        x=f'start_date:T',
        x2='end_date:T'
    )
    
    st.altair_chart(base_chart + anomaly_regions, use_container_width=True)

    st.header("📝 Detected Anomaly Periods")
    if not anomaly_periods.empty:
        for index, row in anomaly_periods.iterrows():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.write(f"**Start:** {row['start_date'].strftime('%Y-%m-%d')}")
            with col2:
                st.write(f"**End:** {row['end_date'].strftime('%Y-%m-%d')}")
            with col3:
                st.write(f"**Severity:** {row['severity']:.2f}")
            with col4:
                if st.button("Explain", key=f"explain_{index}"):
                    st.session_state.explaining_anomaly = row
                    st.session_state.explanation = None # Reset previous explanation
    else:
        st.info("No anomalies were detected with the current settings.")

    # --- Chatbot Explanation Section ---
    if st.session_state.explaining_anomaly is not None:
        st.header("💬 Anomaly Explanation Chat")
        anomaly_to_explain = st.session_state.explaining_anomaly
        start = anomaly_to_explain['start_date'].strftime('%Y-%m-%d')
        end = anomaly_to_explain['end_date'].strftime('%Y-%m-%d')
        
        st.info(f"Let's explain the anomaly from **{start}** to **{end}**.")

        # Display explanation if it exists
        if st.session_state.explanation:
            st.markdown("---")
            st.markdown("### AI-Generated Explanation")
            st.markdown(st.session_state.explanation)
        else:
            # Ask for context
            context_prompt = f"To give you the best explanation, please provide some context about this data. For example: 'Apple Inc. stock price', 'Server CPU usage', or 'Daily user signups'."
            context = st.text_input(context_prompt, key=f"context_{start}")

            if st.button("Get Explanation", key=f"get_exp_{start}"):
                if context:
                    with st.spinner("Asking AI for explanation..."):
                        explanation = get_explanation(anomaly_to_explain['start_date'], anomaly_to_explain['end_date'], context)
                        st.session_state.explanation = explanation
                        st.experimental_rerun() # Rerun to display the explanation
                else:
                    st.warning("Please provide context for a more accurate explanation.")

    st.session_state.run_analysis = False 
elif not st.session_state.get('df', pd.DataFrame()).empty:
     pass # Don't show the initial message if a file is loaded but analysis not run
else:
    st.info("Upload a CSV file and configure the parameters in the sidebar to begin.")
