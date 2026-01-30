"""
DCML Project - Subtle CPU Attack Injector
Creates a WEAK CPU stress that overlaps with normal activity.
This makes it harder for the model to classify, resulting in more realistic accuracy.
"""

import time
import multiprocessing
import math

# Configuration - VERY WEAK attack
DURATION = 300          # 5 minutes
MAX_CPU_WORKERS = 1     # Only 1 worker (very weak)

def light_cpu_burn():
    """Light CPU task with pauses - creates mild stress"""
    while True:
        # Light work
        for _ in range(100000):
            math.sqrt(12345.6789)
        # Pause - creates intermittent pattern
        time.sleep(0.05)

def subtle_cpu_attack():
    print("=" * 50)
    print("SUBTLE CPU ATTACK - Light Stress")
    print("=" * 50)
    print("This creates WEAK CPU stress that overlaps with normal activity.")
    print("Expected CPU increase: only 5-15%")
    print("=" * 50)
    print("\nStarting subtle attack...")
    print("Press Ctrl+C to stop.\n")
    
    workers = []
    start_time = time.time()
    
    try:
        # Start just 1 light worker
        p = multiprocessing.Process(target=light_cpu_burn)
        p.daemon = True
        p.start()
        workers.append(p)
        print("Started 1 light CPU worker")
        
        while time.time() - start_time < DURATION:
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed:3d}s] Attack running (subtle mode)...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n--- Attack stopped by user ---")
    finally:
        for w in workers:
            w.terminate()
        print("Terminated workers")
        print("Attack finished.")

if __name__ == "__main__":
    subtle_cpu_attack()
