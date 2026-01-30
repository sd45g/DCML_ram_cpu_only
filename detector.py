"""
DCML Project - Runtime Anomaly Detector
System Attack Detection (CPU Stress / Memory Leak)

This script monitors your system in real-time and detects attacks
using the pre-trained machine learning model.
Features monitored: CPU and RAM only
"""

import psutil
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_FILE = "anomaly_detector.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES = ["cpu", "ram"]  # Only CPU and RAM
CHECK_INTERVAL = 1  # seconds between checks
ALERT_THRESHOLD = 3  # consecutive anomalies before alerting

# Global state
consecutive_anomalies = 0

def load_model():
    """Load the trained model and scaler"""
    print("Loading anomaly detection model...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Model loaded successfully!")
    return model, scaler

def get_system_status():
    """Collect current system metrics (CPU and RAM only)"""
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent
    }

def predict_anomaly(model, scaler, status):
    """Use the model to predict if current state is anomaly"""
    # Create DataFrame with only CPU and RAM
    features_df = pd.DataFrame([[
        status["cpu"],
        status["ram"]
    ]], columns=FEATURES)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
    else:
        confidence = 100.0
    
    return prediction, confidence

def format_status(status, prediction, confidence):
    """Format status for display"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    status_str = (
        f"CPU: {status['cpu']:5.1f}% | "
        f"RAM: {status['ram']:5.1f}%"
    )
    
    if prediction == 0:
        result = f"[{timestamp}] {status_str} | Status: NORMAL"
    else:
        result = f"[{timestamp}] {status_str} | Status: ANOMALY ({confidence:.0f}%)"
    
    return result

def run_detector():
    """Main detection loop"""
    global consecutive_anomalies
    
    print("\n" + "=" * 60)
    print("   DCML Anomaly Detector - Real-Time Monitor")
    print("=" * 60)
    print()
    
    # Load model
    model, scaler = load_model()
    
    print()
    print("Starting real-time monitoring...")
    print("Press Ctrl+C to stop.")
    print()
    print("-" * 60)
    
    try:
        while True:
            # Get current status
            status = get_system_status()
            
            # Predict
            prediction, confidence = predict_anomaly(model, scaler, status)
            
            # Track consecutive anomalies
            if prediction == 1:
                consecutive_anomalies += 1
            else:
                consecutive_anomalies = 0
            
            # Display status
            print(format_status(status, prediction, confidence))
            
            # Alert if sustained anomaly
            if consecutive_anomalies >= ALERT_THRESHOLD:
                print()
                print("!" * 60)
                print("!!! ALERT: Attack Detected !!!")
                print(f"!!! Anomaly detected for {consecutive_anomalies} seconds")
                print("!" * 60)
                print()
            
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt: 
        print()
        print("-" * 60)
        print("Monitoring stopped.")
        print("=" * 60)

if __name__ == "__main__":
    run_detector()
