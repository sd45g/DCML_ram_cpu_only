import multiprocessing
import time
import sys

def cpu_worker(stop_event):
    """Effect: High CPU usage (Math Calculation)"""
    while not stop_event.is_set():
        _ = 234234 * 234234
        # STEALTH MODE: Sleep briefly to avoid 100% CPU (Target ~50%)
        time.sleep(0.1)

def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting CPU Stress (Crypto Miner Simulation)...")
    stop_event = multiprocessing.Event()
    processes = []
    
    # Use almost all cores
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    for _ in range(num_cores):
        p = multiprocessing.Process(target=cpu_worker, args=(stop_event,))
        p.daemon = True
        p.start()
        processes.append(p)
        
    print(f"[{time.strftime('%H:%M:%S')}] Running on {num_cores} cores. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
