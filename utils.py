import streamlit as st
import numpy as np
import requests
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from models import LSTMAutoencoder # Import the class from models.py

# --- Helper Functions for Anomaly Detection ---

def calculate_benford_deviation(window):
    """Calculates the Chi-Squared statistic for Benford's Law deviation."""
    if not isinstance(window, np.ndarray):
        window = np.array(window)
        
    first_digits = np.array([int(str(v)[0]) for v in window if v > 0])
    if len(first_digits) < 10:
        return 0

    observed_counts = np.bincount(first_digits, minlength=10)[1:]
    n = len(first_digits)
    expected_proportions = np.array([0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046])
    expected_counts = n * expected_proportions
    
    expected_counts[expected_counts == 0] = 1e-9
    chi_squared = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    return chi_squared

def normalize_scores(scores):
    """Normalizes scores to have a mean of 0 and a standard deviation of 1."""
    scores = np.array(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores)
    if std_dev == 0:
        return np.zeros_like(scores)
    return (scores - mean) / std_dev

def create_sequences(values, window_size):
    """Creates overlapping sequences for LSTM model."""
    output = []
    for i in range(len(values) - window_size + 1):
        output.append(values[i : (i + window_size)])
    return np.stack(output)

def detect_anomalies(df, value_col, window_size, threshold_multiplier, epochs):
    """Main function to detect anomalies using the hybrid LSTM model."""
    values = df[value_col].to_numpy()
    
    # --- LSTM Part: Calculate Reconstruction Error ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))
    
    X = create_sequences(scaled_values, window_size)
    X_tensor = torch.from_numpy(X).float()
    
    lstm_model = LSTMAutoencoder(seq_len=window_size, n_features=1)
    criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    # Use the user-defined number of epochs
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        outputs = lstm_model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        X_pred_tensor = lstm_model(X_tensor)
    reconstruction_errors = np.mean(np.abs(X_pred_tensor.numpy() - X), axis=1).flatten()

    # --- Benford's Law Part: Calculate Deviation Score ---
    benford_scores = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        benford_scores.append(calculate_benford_deviation(window))
    
    # --- Combine Scores and add to DataFrame ---
    norm_reconstruction = normalize_scores(reconstruction_errors)
    norm_benford = normalize_scores(benford_scores)
    
    combined_scores = norm_reconstruction + norm_benford
    
    # Pad all scores to match original data length for DataFrame columns
    df['reconstruction_score'] = np.pad(norm_reconstruction, (window_size - 1, 0), 'constant', constant_values=0)
    df['benford_score'] = np.pad(norm_benford, (window_size - 1, 0), 'constant', constant_values=0)
    df['anomaly_score'] = np.pad(combined_scores, (window_size - 1, 0), 'constant', constant_values=0)

    # Identify anomalies based on the threshold
    threshold = np.mean(combined_scores) + threshold_multiplier * np.std(combined_scores)
    df['is_anomaly'] = df['anomaly_score'] > threshold
    
    return df, threshold

def get_anomaly_periods(df, date_col):
    """Groups consecutive anomalies into periods and calculates feature contribution."""
    df['anomaly_group'] = (df['is_anomaly'] != df['is_anomaly'].shift()).cumsum()
    
    # Aggregate to find periods and average component scores
    anomaly_periods = df[df['is_anomaly']].groupby('anomaly_group').agg(
        start_date=(date_col, 'min'),
        end_date=(date_col, 'max'),
        severity=('anomaly_score', 'max'),
        reconstruction_contribution=('reconstruction_score', 'mean'),
        benford_contribution=('benford_score', 'mean')
    ).reset_index(drop=True)
    
    return anomaly_periods

def get_explanation(start_date, end_date, context):
    """Calls the Gemini API to get an explanation for an anomaly."""
    prompt = f"""For a time-series dataset representing "{context}", an anomaly was detected between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}. Search for major news or events (such as financial reports, market changes, or world events) within this period that could explain this anomaly. Provide a brief, bulleted summary."""
    
    try:
        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        api_key = st.secrets["GEMINI_API_KEY"]
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("candidates") and result["candidates"][0].get("content", {}).get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Could not retrieve a valid explanation from the AI model."
            
    except requests.exceptions.RequestException as e:
        return f"Could not retrieve explanation due to a network error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
