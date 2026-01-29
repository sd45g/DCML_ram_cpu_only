import psutil
import time
import csv
import os

# 1. Define where to save the data
DATA_FILE = "my_system_data.csv"

last_bytes_sent = psutil.net_io_counters().bytes_sent
def get_system_status():
    global last_bytes_sent
    
    # Get current totals
    current_bytes_sent = psutil.net_io_counters().bytes_sent
    
    # Calculate how many bytes were sent ONLY in the last 1 second
    net_speed = current_bytes_sent - last_bytes_sent
    
    # Update the global variable for next time
    last_bytes_sent = current_bytes_sent

    # Count total threads across all processes (will spike during Socket Drain)
    total_threads = sum(p.info['num_threads'] for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
    
    return {
        "timestamp": time.time(),
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "sockets": len(psutil.net_connections()),
        "net_speed": net_speed,
        "threads": total_threads,
        "label": 1  # Anomaly behavior
    }
def start_monitoring():
    # Check if file exists so we don't write the header twice
    file_exists = os.path.isfile(DATA_FILE)
    
    with open(DATA_FILE, "a", newline="") as f:
        # Get the keys from our status function for the CSV header
        fieldnames = ["timestamp", "cpu", "ram", "sockets", "net_speed", "threads", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        print("--- MONITORING STARTED ---")
        print("I am recording your laptop's health every 1 second.")
        print("Press Ctrl+C to stop when you are done.")
        
        try:
            while True:
                stats = get_system_status()
                writer.writerow(stats)
                f.flush() # This ensures the data is saved even if the PC crashes
                print(f"Recorded: CPU {stats['cpu']}% | Sockets: {stats['sockets']} | Threads: {stats['threads']}")
                time.sleep(1) # Wait for 1 second before the next check
        except KeyboardInterrupt:
            print("\n--- MONITORING STOPPED ---")

if __name__ == "__main__":
    start_monitoring()