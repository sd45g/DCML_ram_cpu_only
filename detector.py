import joblib
import time
import os
import sys
import numpy as np
import pandas as pd
from monitor import SystemMonitor

# Configuration
MODEL_FILE = "anomaly_detector.pkl"
SCALER_FILE = "scaler.pkl"

def main():
    print("\n" + "="*60)
    print("LIVE ANOMALY DETECTOR")
    print("="*60)
    
    # 1. Load Brain
    if not os.path.exists(MODEL_FILE):
        print(f"[!] Error: {MODEL_FILE} not found. Run train_model.py first!")
        return
        
    print("Loading model...", end="")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(" Done!")
    print(f"Algorithm: {type(model).__name__}")
    
    # 2. Init Monitor
    monitor = SystemMonitor()
    print("\nStarting Guard... (Press Ctrl+C to stop)")
    print("-" * 60)
    
    # 3. Detection Loop
    try:
        # Initial wait for rate calculation
        time.sleep(1)
        
        while True:
            # Get Data
            metrics = monitor.get_metrics()
            
            # Prepare for Model (must match training order)
            feature_names = ['cpu', 'ram', 'processes', 'disk_read', 'disk_write', 'net_sent']
            features_df = pd.DataFrame([[
                metrics['cpu'],
                metrics['ram'],
                metrics['processes'],
                metrics['disk_read'],
                metrics['disk_write'],
                metrics['net_sent']
            ]], columns=feature_names)
            
            # Scale
            features_scaled = scaler.transform(features_df)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            # Try to get probability if supported
            try:
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[prediction] * 100
            except:
                confidence = 100.0
            
            # Display
            # Clear line/overwrite? We'll just print new lines for log history
            timestamp = time.strftime("%H:%M:%S")
            
            if prediction == 1:
                status = "⚠️  ATTACK DETECTED!"
                color = "\033[91m" # Red (if terminal supports it)
            else:
                status = "✅ Normal"
                color = "\033[92m" # Green
                
            print(f"[{timestamp}] {status:<20} | Conf: {confidence:.0f}% | "
                  f"CPU: {metrics['cpu']:.0f}% RAM: {metrics['ram']:.0f}%")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nDetector Stopped.")

if __name__ == "__main__":
    main()
