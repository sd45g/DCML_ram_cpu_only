"""
DCML Project - Runtime Anomaly Detector
Socket Drain Attack Detection System

This script monitors your system in real-time and detects Socket Drain attacks
using the pre-trained machine learning model.
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
FEATURES = ["cpu", "ram", "sockets", "net_speed", "threads"]
CHECK_INTERVAL = 1  # seconds between checks
ALERT_THRESHOLD = 3  # consecutive anomalies before alerting

# Global state
last_bytes_sent = psutil.net_io_counters().bytes_sent
consecutive_anomalies = 0

def load_model():
    """Load the trained model and scaler"""
    print("Loading anomaly detection model...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Model loaded successfully!")
    return model, scaler

def get_system_status():
    """Collect current system metrics"""
    global last_bytes_sent
    
    # Network speed (delta)
    current_bytes_sent = psutil.net_io_counters().bytes_sent
    net_speed = current_bytes_sent - last_bytes_sent
    last_bytes_sent = current_bytes_sent
    
    # Thread count
    total_threads = sum(
        p.info['num_threads'] 
        for p in psutil.process_iter(['num_threads']) 
        if p.info['num_threads']
    )
    
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "sockets": len(psutil.net_connections()),
        "net_speed": net_speed,
        "threads": total_threads
    }

def predict_anomaly(model, scaler, status):
    """Use the model to predict if current state is anomaly"""
    # Create DataFrame with proper feature names
    features_df = pd.DataFrame([[
        status["cpu"],
        status["ram"],
        status["sockets"],
        status["net_speed"],
        status["threads"]
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
        f"RAM: {status['ram']:5.1f}% | "
        f"Sockets: {status['sockets']:5d} | "
        f"Threads: {status['threads']:5d}"
    )
    
    if prediction == 0:
        result = f"[{timestamp}] {status_str} | Status: ✓ NORMAL"
    else:
        result = f"[{timestamp}] {status_str} | Status: ⚠ ANOMALY ({confidence:.0f}%)"
    
    return result

def run_detector():
    """Main detection loop"""
    global consecutive_anomalies
    
    print("\n" + "=" * 70)
    print("   DCML Socket Drain Anomaly Detector - Runtime Monitor")
    print("=" * 70)
    print()
    
    # Load model
    model, scaler = load_model()
    
    print()
    print("Starting real-time monitoring...")
    print("Press Ctrl+C to stop.")
    print()
    print("-" * 70)
    
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
                print("!" * 70)
                print("!!! ALERT: Socket Drain Attack Detected !!!")
                print(f"!!! Sustained anomaly for {consecutive_anomalies} seconds")
                print(f"!!! Sockets: {status['sockets']} (normal: ~2000)")
                print("!" * 70)
                print()
            
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt: 
        print()
        print("-" * 70)
        print("Monitoring stopped.")
        print("=" * 70)

if __name__ == "__main__":
    run_detector()
