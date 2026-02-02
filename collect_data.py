import csv
import time
import subprocess
import os
import sys
from monitor import SystemMonitor

# Configuration
DATA_FILE = "my_system_data.csv"
SAMPLES_NORMAL = 600  # 300 seconds (5 minutes) - Better for "Realistic Accuracy"
SAMPLES_ATTACK = 150  # 75 seconds per attack
INTERVAL = 0.5

def collect_phase(writer, monitor, label, samples, description, injector_script=None):
    print(f"\n" + "="*60)
    print(f"PHASE: {description}")
    print(f"Collecting {samples} samples...")
    print("="*60)
    
    injector_process = None
    
    # Start Injector if defined
    if injector_script:
        print(f"Launching injector: {injector_script}...")
        injector_process = subprocess.Popen([sys.executable, injector_script], 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL)
        # Give it a moment to ramp up
        time.sleep(2)
        
    try:
        for i in range(samples):
            # 1. Get Metrics
            metrics = monitor.get_metrics()
            
            # 2. Add Label
            metrics["label"] = label
            
            # 3. Save to file
            writer.writerow(metrics)
            
            # 4. Print status
            print(f"\r[{i+1}/{samples}] CPU: {metrics['cpu']}% | RAM: {metrics['ram']}% | Label: {label}", end="")
            
            # 5. Wait
            time.sleep(INTERVAL)
            
    finally:
        # Stop Injector
        if injector_process:
            print(f"\nStopping injector...")
            injector_process.terminate()
            injector_process.wait()
            # Cool down to let system recover
            print("Cooling down (5s)...")
            time.sleep(5)

def main():
    print("Initializing Data Collection...")
    monitor = SystemMonitor()
    
    # Open CSV for writing
    with open(DATA_FILE, "w", newline="") as f:
        fieldnames = ["cpu", "ram", "processes", "disk_read", "disk_write", "net_sent", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Phase 1: Normal
        collect_phase(writer, monitor, 0, SAMPLES_NORMAL, "Normal Behavior (Please browse web/code)", None)
        
        # Phase 2: CPU Attack
        collect_phase(writer, monitor, 1, SAMPLES_ATTACK, "CPU Attack (Crypto Miner)", "injectors/cpu_stress.py")
        
        # Phase 3: Memory Attack
        collect_phase(writer, monitor, 1, SAMPLES_ATTACK, "Memory Attack (Leak)", "injectors/memory_leak.py")
        
        # Phase 4: Network Attack
        collect_phase(writer, monitor, 1, SAMPLES_ATTACK, "Network Attack (Data Exfiltration)", "injectors/network_flood.py")
        
        # Phase 5: Process Attack
        collect_phase(writer, monitor, 1, SAMPLES_ATTACK, "Process Attack (Malware Rep)", "injectors/process_bomb.py")
        
    print("\n" + "="*60)
    print("Data Collection Complete!")
    print(f"File saved: {DATA_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
