"""
DCML Project - Normal Behavior Monitor
Records CPU and RAM when system is operating normally (label=0)
"""

import psutil
import time
import csv
import os

DATA_FILE = "my_system_data.csv"

def get_system_status():
    """Collect current CPU and RAM usage"""
    return {
        "timestamp": time.time(),
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "label": 0  # Normal behavior
    }

def start_monitoring():
    file_exists = os.path.isfile(DATA_FILE)
    
    with open(DATA_FILE, "a", newline="") as f:
        fieldnames = ["timestamp", "cpu", "ram", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        print("--- MONITORING STARTED ---")
        print("Recording CPU and RAM every 1 second.")
        print("Press Ctrl+C to stop when you are done.")
        
        try:
            while True:
                stats = get_system_status()
                writer.writerow(stats)
                f.flush()
                print(f"Recorded: CPU {stats['cpu']:.1f}% | RAM {stats['ram']:.1f}%")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n--- MONITORING STOPPED ---")

if __name__ == "__main__":
    start_monitoring()