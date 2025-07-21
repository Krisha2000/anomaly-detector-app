import streamlit as st
import pandas as pd
import numpy as np
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

st.title("🤖 Hybrid Time-Series Anomaly Detector")
st.markdown("This tool uses a Hybrid LSTM Autoencoder combined with Benford's Law to detect anomalies. Upload your data to begin.")

# Initialize session state for chat
if 'explaining_anomaly' not in st.session_state:
    st.session_state.explaining_anomaly = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not PYTORCH_AVAILABLE:
        st.error("PyTorch not found! Please ensure it is listed in your requirements.txt for deployment.")

    uploaded_file = st.file_uploader("Upload your time-series CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success("File uploaded successfully!")
        
        st.subheader("Column Selection")
        date_col = st.selectbox("Select Date/Time Column", df.columns, index=0)
        value_col = st.selectbox("Select Value Column (Target)", df.columns, index=1 if len(df.columns) > 1 else 0)
        
        st.subheader("Model Parameters")
        st.info("The app uses the Hybrid LSTM model.")
        
        window_size = st.number_input("Window Size / Look-back Period", min_value=10, max_value=200, value=30, step=5)
        threshold = st.number_input("Anomaly Threshold (Std Dev)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

        if st.button("🚀 Detect Anomalies", use_container_width=True):
            st.session_state.date_col = date_col
            st.session_state.value_col = value_col
            st.session_state.window_size = window_size
            st.session_state.threshold = threshold
            st.session_state.run_analysis = True
            # Reset chat state on new analysis
            st.session_state.explaining_anomaly = None
            st.session_state.explanation = None

# --- Main Content Area ---
if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    with st.spinner("Analyzing data... The Hybrid LSTM model may take a moment to train."):
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
            st.session_state.threshold
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

        if st.session_state.explanation:
            st.markdown("---")
            st.markdown("### AI-Generated Explanation")
            st.markdown(st.session_state.explanation)
        else:
            context_prompt = f"To give you the best explanation, please provide some context about this data. For example: 'Apple Inc. stock price', 'Server CPU usage', or 'Daily user signups'."
            context = st.text_input(context_prompt, key=f"context_{start}")

            if st.button("Get Explanation", key=f"get_exp_{start}"):
                if context:
                    with st.spinner("Asking AI for explanation..."):
                        explanation = get_explanation(anomaly_to_explain['start_date'], anomaly_to_explain['end_date'], context)
                        st.session_state.explanation = explanation
                        st.rerun() 
                else:
                    st.warning("Please provide context for a more accurate explanation.")

    st.session_state.run_analysis = False 
elif not st.session_state.get('df', pd.DataFrame()).empty:
     pass 
else:
    st.info("Upload a CSV file and configure the parameters in the sidebar to begin.")


# --- Explainer Section ---
with st.expander("💡 How the Hybrid LSTM Model Works"):
    st.markdown("""
    Our model combines two different ways of thinking to find anomalies. It looks at both the *shape* of the data and the statistical *properties* of the numbers themselves.
    """)

    st.subheader("Step 1: Input Data Window")
    st.markdown("The model doesn't look at the whole dataset at once. Instead, it slides a 'window' across the data to analyze small, sequential chunks.")
    
    sample_data = pd.DataFrame({
        'Day': pd.to_datetime(pd.date_range('2023-01-01', periods=30)),
        'Value': [10, 12, 11, 13, 15, 14, 16, 18, 20, 19, 21, 23, 22, 25, 45, 48, 28, 26, 27, 29, 30, 32, 31, 33, 35, 34, 36, 38, 37, 40]
    })
    input_chart = alt.Chart(sample_data).mark_line(point=True).encode(
        x=alt.X('Day:T', title='Time'),
        y=alt.Y('Value:Q', title='Value'),
        tooltip=['Day', 'Value']
    ).properties(title='A Sample Time-Series Window')
    st.altair_chart(input_chart, use_container_width=True)

    st.subheader("Step 2: Parallel Analysis")
    st.markdown("Each window of data is analyzed by two different methods at the same time:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🧠 **LSTM Autoencoder Analysis**")
        st.markdown("""
        - **What it does:** Learns the normal *shape* and *pattern* of the data.
        - **How it finds anomalies:** It tries to reconstruct the input window. If the window has an unusual pattern, the model struggles to rebuild it accurately, resulting in a high **Reconstruction Error**.
        """)

    with col2:
        st.info("📊 **Benford's Law Analysis**")
        st.markdown("""
        - **What it does:** Checks if the first digits of the numbers in the window follow a natural statistical distribution.
        - **How it finds anomalies:** If the digits are not distributed naturally (e.g., too many 9s), it suggests the data might be manipulated or unnatural, resulting in a high **Benford's Deviation Score**.
        """)

    st.subheader("Step 3: Combine Scores for Final Decision")
    st.markdown("""
    The scores from both analyses are normalized and added together. An event is only flagged as a **high-confidence anomaly** if it's suspicious from *both* a pattern perspective and a statistical perspective.
    
    > **Anomaly Score** = (LSTM Reconstruction Error) + (Benford's Deviation Score)
    
    This hybrid approach reduces false positives and finds more meaningful anomalies.
    """)

    st.subheader("Autoencoder Architecture")
    st.image("https://i.imgur.com/2u9zP5G.png", caption="The LSTM Autoencoder compresses data to its essential features (encoding) and then tries to rebuild it (decoding).")
