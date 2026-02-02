import time
import sys

def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting Memory Leak Simulation...")
    
    data = []
    try:
        while True:
            # TUNED UPDATED: Allocated 50MB (was 5MB).
            # Reason: Your "Normal" RAM is high (81%), so we need a stronger leak to be visible.
            chunk = bytearray(50 * 1024 * 1024) 
            data.append(chunk)
            
            # Print current Usage
            size_gb = (len(data) * 50) / 1024
            print(f"\rAllocated: {size_gb:.2f} GB", end="")
            
            time.sleep(0.5)
            
    except MemoryError:
        print("\n[!] Memory Full! Stopping to prevent crash.")
    except KeyboardInterrupt:
        print("\nStopping...")
        data.clear()

if __name__ == "__main__":
    main()
