# Hybrid Time-Series Anomaly Detector

This project is an interactive web application built with Streamlit that detects anomalies in time-series data. It features an innovative hybrid model combining a PyTorch-based LSTM Autoencoder with statistical analysis using Benford's Law. The application is further enhanced with an AI-powered chatbot that provides real-world context for detected anomalies.

**[➡️ View the Live Application Here](https://hybrid-anomaly-detector-app-krisha-sompura.streamlit.app/)**

---

## Core Concept: Why a Hybrid Model?

Traditional anomaly detection models often focus on a single aspect of the data. For instance, a deep learning model might be excellent at identifying unusual shapes and patterns, while a statistical model might be better at finding data that looks unnatural or manipulated. This project combines both approaches to create a more robust and intelligent detection system.

1. **Deep Learning Path (LSTM Autoencoder):** This part of the model learns the normal temporal "rhythm" and patterns of the data. It finds anomalies by identifying sequences that it cannot reconstruct accurately, resulting in a high **Reconstruction Error**. This is effective for spotting unusual shapes, spikes, or dips.


<img width="1339" height="437" alt="image" src="https://github.com/user-attachments/assets/51e2bd47-e4a8-48e2-8264-ae69a262f909" />


2. **Statistical Path (Benford's Law):** This part of the model acts as a "forensic accountant." It checks if the first digits of the numbers in the data follow a natural distribution. Data that has been manipulated or is not naturally occurring often violates this law, resulting in a high **Benford's Deviation Score**.


    <img width="816" height="452" alt="image" src="https://github.com/user-attachments/assets/0697c004-e887-4876-ab5c-cc5d83704f0e" />



By combining these two scores, the model only flags high-confidence anomalies that are both pattern-wise unusual *and* statistically unnatural, significantly reducing false positives.

## Features

* **Interactive UI:** A clean and professional interface built with Streamlit.

* **Dynamic Data Upload:** Upload any time-series data in CSV format.

* **Flexible Configuration:** Dynamically select the date and value columns from your data and configure model parameters like window size, training epochs, and anomaly sensitivity.

* **Advanced Hybrid Model:** Utilizes a real PyTorch LSTM Autoencoder combined with Benford's Law for robust detection.

* **AI-Powered Explanations:** A built-in chatbot, powered by the Gemini API, that allows you to ask for real-world context (news, financial events, etc.) behind any detected anomaly.

* **Rich Visualizations:** An interactive chart displays the time-series with highlighted anomalies, and a second chart visualizes the anomaly score over time.

* **Built-in Documentation:** The app includes an expandable section that visually explains the model's architecture and workflow.



## Project Structure

The project is organized into four main files for clarity and maintainability:

* **`app.py`**: The main application file that handles the Streamlit user interface and coordinates the overall workflow.

* **`utils.py`**: A utility module that contains all the core logic for data processing, anomaly detection, and communication with the Gemini API.

* **`models.py`**: Defines the PyTorch `LSTMAutoencoder` neural network class.

* **`requirements.txt`**: Lists all the necessary Python libraries for the project.

## Setup and Installation

To run this application locally, please follow these steps.

### 1. Create a Virtual Environment

It is highly recommended to create an isolated Python environment to avoid conflicts with other projects.

- Create the environment
**`python -m venv venv`**

- Activate it (on Windows)
**`.\venv\Scripts\activate`**

- Activate it (on macOS/Linux)
**`source venv/bin/activate`**


### 2. Install Dependencies

- Install all the required libraries using the `requirements.txt` file.
**`pip install -r requirements.txt`**


### 3. Configure API Key

The application uses the Gemini API for the explanation feature. To use it, you need to provide an API key.

* Create a folder in your project directory named `.streamlit`.

* Inside that folder, create a file named `secrets.toml`.

* Add your API key to this file as follows:
**`GEMINI_API_KEY = "YOUR_API_KEY_GOES_HERE"`**


## How to Run the Application

Once you have completed the setup, you can run the application with a single command from your terminal:
**`streamlit run app.py`**


The application will open in a new tab in your web browser.

## Technologies Used

* **Backend & UI:** Streamlit

* **Deep Learning:** PyTorch

* **Data Manipulation:** Pandas, NumPy, Scikit-learn

* **Data Visualization:** Altair, Plotly

* **AI Explanations:** Google Gemini API
