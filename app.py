import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Plotly Architecture Diagram Function ---
def create_architecture_diagram():
    fig = go.Figure()
    y_center, box_height, box_width = 0.5, 0.35, 0.6
    blocks = {
        "Input": {"x": 0, "y": y_center, "title": "Input Window", "subtitle": "Shape: [30, 1]"},
        "Encoder_Top": {"x": 1.5, "y": y_center + 0.22, "title": "LSTM Layer", "subtitle": "128 Neurons"},
        "Encoder_Bottom": {"x": 1.5, "y": y_center - 0.22, "title": "LSTM Layer", "subtitle": "64 Neurons"},
        "Latent": {"x": 3, "y": y_center, "title": "Latent Vector", "subtitle": "Compressed"},
        "Decoder_Top": {"x": 4.5, "y": y_center + 0.22, "title": "LSTM Layer", "subtitle": "64 Neurons"},
        "Decoder_Bottom": {"x": 4.5, "y": y_center - 0.22, "title": "LSTM Layer", "subtitle": "128 Neurons"},
        "Output": {"x": 6, "y": y_center, "title": "Output Window", "subtitle": "Reconstructed"},
    }
    for name, b in blocks.items():
        fig.add_shape(type="rect", x0=b['x'] - box_width/2, y0=b['y'] - box_height/2, x1=b['x'] + box_width/2, y1=b['y'] + box_height/2,
                      line=dict(color="#d1d5db"), fillcolor="#f9fafb")
        fig.add_annotation(x=b['x'], y=b['y']+0.06, text=f"<b>{b['title']}</b>", showarrow=False, font=dict(color="#1f2937", size=12))
        fig.add_annotation(x=b['x'], y=b['y']-0.06, text=b['subtitle'], showarrow=False, font=dict(color="#4b5563", size=10))
    fig.add_shape(type="rect", x0=1.5 - 0.45, y0=y_center - 0.55, x1=1.5 + 0.45, y1=y_center + 0.55, line=dict(color="#3b82f6", width=2, dash="dash"))
    fig.add_annotation(x=1.5, y=y_center + 0.65, text="<b>ENCODER</b>", showarrow=False, font=dict(color="#3b82f6", size=13))
    fig.add_shape(type="rect", x0=4.5 - 0.45, y0=y_center - 0.55, x1=4.5 + 0.45, y1=y_center + 0.55, line=dict(color="#10b981", width=2, dash="dash"))
    fig.add_annotation(x=4.5, y=y_center + 0.65, text="<b>DECODER</b>", showarrow=False, font=dict(color="#10b981", size=13))
    def draw_arrow(x_start, x_end, y_pos):
        fig.add_annotation(x=x_end, y=y_pos, ax=x_start, ay=y_pos, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#6b7280")
    draw_arrow(blocks['Input']['x'] + box_width/2, blocks['Encoder_Top']['x'] - 0.45, y_center)
    draw_arrow(blocks['Encoder_Top']['x'] + 0.45, blocks['Latent']['x'] - box_width/2, y_center)
    draw_arrow(blocks['Latent']['x'] + box_width/2, blocks['Decoder_Top']['x'] - 0.45, y_center)
    draw_arrow(blocks['Decoder_Top']['x'] + 0.45, blocks['Output']['x'] - box_width/2, y_center)
    fig.add_annotation(x=1.5, y=blocks['Encoder_Bottom']['y'] + box_height/2, ax=1.5, ay=blocks['Encoder_Top']['y'] - box_height/2, showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#6b7280")
    fig.add_annotation(x=4.5, y=blocks['Decoder_Bottom']['y'] + box_height/2, ax=4.5, ay=blocks['Decoder_Top']['y'] - box_height/2, showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#6b7280")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", xaxis=dict(range=[-0.5, 6.5], visible=False), yaxis=dict(range=[-0.2, 1.2], visible=False), height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig

# --- Streamlit UI ---

st.title("Hybrid Time-Series Anomaly Detector")
st.markdown("This tool utilizes a Hybrid LSTM Autoencoder combined with Benford's Law to detect anomalies. Upload your data to begin.")

# Initialize session state
if 'explaining_anomaly' not in st.session_state:
    st.session_state.explaining_anomaly = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
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
        
        epochs = st.number_input("Training Epochs", min_value=5, max_value=50, value=10, step=5, help="Number of training cycles for the LSTM model. More epochs can lead to better pattern learning but take longer.")
        
        window_size = st.number_input("Window Size", min_value=10, max_value=200, value=30, step=5, help="The number of data points the model looks at in one go.")
        threshold = st.number_input("Anomaly Threshold (Std Dev)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, help="How many standard deviations above the average score to flag an anomaly. Higher is less sensitive.")

        if st.button("Detect Anomalies", use_container_width=True):
            st.session_state.date_col = date_col
            st.session_state.value_col = value_col
            st.session_state.window_size = window_size
            st.session_state.threshold = threshold
            st.session_state.epochs = epochs
            st.session_state.run_analysis = True
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
            
        results_df, threshold_value = detect_anomalies(
            df, 
            value_col, 
            st.session_state.window_size, 
            st.session_state.threshold,
            st.session_state.epochs
        )
        
        if results_df is not None:
            anomaly_periods = get_anomaly_periods(results_df, date_col)
            st.session_state.results_df = results_df
            st.session_state.anomaly_periods = anomaly_periods
            st.session_state.threshold_value = threshold_value
            st.session_state.display_results = True
        else:
            st.session_state.display_results = False

if st.session_state.get('display_results', False):
    st.header("Analysis Results")
    results_df = st.session_state.results_df
    anomaly_periods = st.session_state.anomaly_periods
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    threshold_value = st.session_state.threshold_value

    base_chart = alt.Chart(results_df).mark_line().encode(
        x=alt.X(f'{date_col}:T', title='Date'),
        y=alt.Y(f'{value_col}:Q', title='Value'),
        tooltip=[date_col, value_col]
    ).interactive()
    anomaly_regions = alt.Chart(anomaly_periods).mark_rect(opacity=0.3, color='red').encode(x=f'start_date:T', x2='end_date:T')
    st.altair_chart(base_chart + anomaly_regions, use_container_width=True)

    st.subheader("Anomaly Score Over Time")
    score_chart = alt.Chart(results_df).mark_area(
        line={'color':'#3b82f6'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#3b82f6', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X(f'{date_col}:T', title='Date'),
        y=alt.Y('anomaly_score:Q', title='Anomaly Score'),
        tooltip=[date_col, 'anomaly_score']
    ).interactive()
    
    threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold_value]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='threshold:Q')
    st.altair_chart(score_chart + threshold_line, use_container_width=True)

    st.header("Detected Anomaly Periods")
    if not anomaly_periods.empty:
        for index, row in anomaly_periods.iterrows():
            # --- This is the NEW, corrected code ---
            recon_contrib = abs(row['reconstruction_contribution'])
            benford_contrib = abs(row['benford_contribution'])
            total_contrib = recon_contrib + benford_contrib

            if total_contrib > 0:
                recon_percent = int((recon_contrib / total_contrib) * 100)
                benford_percent = 100 - recon_percent
                importance_text = f"LSTM: **{recon_percent}%**, Benford's Law: **{benford_percent}%**"
            else:
                importance_text = "N/A"

            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Period:** {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}")
                st.write(f"**Severity:** {row['severity']:.2f}")
                st.write(f"**Primary Drivers:** {importance_text}")
            with col2:
                if st.button("Explain Anomaly", key=f"explain_{index}"):
                    st.session_state.explaining_anomaly = row
                    st.session_state.explanation = None
    else:
        st.info("No anomalies were detected with the current settings.")

    if st.session_state.explaining_anomaly is not None:
        st.header("Anomaly Explanation")
        anomaly_to_explain = st.session_state.explaining_anomaly
        start = anomaly_to_explain['start_date'].strftime('%Y-%m-%d')
        end = anomaly_to_explain['end_date'].strftime('%Y-%m-%d')
        
        st.info(f"Explaining the anomaly from **{start}** to **{end}**.")

        if st.session_state.explanation:
            st.markdown("### AI-Generated Explanation")
            st.markdown(st.session_state.explanation)
        else:
            context_prompt = f"To provide a relevant explanation, please specify the context of this data (e.g., 'Apple Inc. stock price', 'Server CPU usage')."
            context = st.text_input(context_prompt, key=f"context_{start}")

            if st.button("Get Explanation", key=f"get_exp_{start}"):
                if context:
                    with st.spinner("Generating explanation..."):
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

with st.expander("How the Hybrid LSTM Model Works"):
    st.markdown("""
    This model combines two distinct analytical methods to identify anomalies. It evaluates both the temporal **shape** of the data and the statistical **properties** of the numbers themselves.
    """)
    st.subheader("Autoencoder Architecture")
    st.plotly_chart(create_architecture_diagram(), use_container_width=True)
    st.subheader("Step 1: Input Data Window")
    st.markdown("The model analyzes the dataset by sliding a 'window' across the data, examining small, sequential chunks rather than the entire series at once.")
    st.subheader("Step 2: Parallel Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**LSTM Autoencoder Analysis**")
        st.markdown("""
        - **Objective:** To learn the normal temporal patterns and shape of the data.
        - **Method:** The model attempts to reconstruct the input window. A high **Reconstruction Error** occurs if the pattern is unusual, signaling a potential anomaly.
        """)
    with col2:
        st.info("**Benford's Law Analysis**")
        st.markdown("""
        - **Objective:** To verify if the numbers follow a natural statistical distribution.
        - **Method:** The distribution of the first digits in the window is checked. A high **Benford's Deviation Score** suggests the data may be unnatural or manipulated.
        """)
    st.subheader("Step 3: Combined Scoring")
    st.markdown("""
    The scores from both analyses are normalized and aggregated. An event is flagged as a high-confidence anomaly only if it is suspicious from both a pattern and a statistical perspective.
    > **Final Anomaly Score** = (Normalized Reconstruction Error) + (Normalized Deviation Score)
    This hybrid methodology is designed to reduce false positives and identify more significant anomalies.
    """)
