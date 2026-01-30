"""
DCML Project - CPU Stress Attack Injector
Simulates gradual CPU stress (like crypto-mining malware)

This attack SLOWLY increases CPU usage over time.
- Harder to detect with simple thresholds
- ML is needed to detect the gradual trend
"""

import time
import multiprocessing
import math

# Configuration
DURATION = 300          # Total attack duration (5 minutes)
RAMP_UP_TIME = 180      # Time to reach max intensity (3 minutes)
MAX_CPU_WORKERS = 4     # Max number of CPU-burning processes

def cpu_burn():
    """CPU-intensive task that burns CPU cycles"""
    while True:
        # Heavy math operations
        for _ in range(1000000):
            math.sqrt(12345.6789) * math.sin(0.5) * math.cos(0.3)

def gradual_cpu_attack():
    print("=" * 50)
    print("CPU STRESS ATTACK - Gradual Increase")
    print("=" * 50)
    print(f"Duration: {DURATION} seconds")
    print(f"Ramp-up time: {RAMP_UP_TIME} seconds")
    print("=" * 50)
    print("\nStarting attack... CPU will gradually increase.")
    print("Run monitor_anomaly.py in another terminal!")
    print("Press Ctrl+C to stop.\n")
    
    workers = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < DURATION:
            elapsed = time.time() - start_time
            
            # Calculate how many workers should be active based on time
            if elapsed < RAMP_UP_TIME:
                # Gradually increase workers
                target_workers = int((elapsed / RAMP_UP_TIME) * MAX_CPU_WORKERS)
            else:
                # Stay at max
                target_workers = MAX_CPU_WORKERS
            
            # Add workers if needed
            while len(workers) < target_workers:
                p = multiprocessing.Process(target=cpu_burn)
                p.daemon = True
                p.start()
                workers.append(p)
                print(f"[{int(elapsed):3d}s] Added CPU worker (total: {len(workers)})")
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n--- Attack stopped by user ---")
    finally:
        # Clean up workers
        for w in workers:
            w.terminate()
        print(f"Terminated {len(workers)} CPU workers")
        print("Attack finished.")

if __name__ == "__main__":
    gradual_cpu_attack()
