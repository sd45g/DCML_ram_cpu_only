import multiprocessing
import time
import os

def sleepy_process():
    """Does nothing, just exists to increase process count"""
    try:
        while True:
            time.sleep(10)
    except:
        pass

def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting Process Bomb (Safe Replication)...")
    
    processes = []
    MAX_PROCESSES = 30 # STEALTH: Only 30 processes (was 200). Hard to spot in 500+ list.
    
    try:
        for i in range(MAX_PROCESSES):
            p = multiprocessing.Process(target=sleepy_process)
            p.daemon = True
            p.start()
            processes.append(p)
            
            if i % 10 == 0:
                print(f"\rSpawned {i} processes...", end="")
            
            # Spawn fast!
            time.sleep(0.01)
            
        print(f"\n[!] Reached limit ({MAX_PROCESSES}). Holding...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
