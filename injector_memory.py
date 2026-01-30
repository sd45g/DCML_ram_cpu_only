"""
DCML Project - Memory Leak Attack Injector  
Simulates a gradual memory leak (like buggy software)

This attack SLOWLY consumes more RAM over time.
- Harder to detect with simple thresholds
- ML is needed to detect the gradual trend
"""

import time
import sys

# Configuration
DURATION = 300              # Total attack duration (5 minutes)
CHUNK_SIZE_MB = 50          # Memory to allocate each step
INTERVAL = 30               # Seconds between allocations
MAX_MEMORY_MB = 500         # Maximum memory to consume

# Store allocated memory so it doesn't get garbage collected
memory_chunks = []

def allocate_memory(size_mb):
    """Allocate a chunk of memory and hold onto it"""
    # Create a bytearray of the specified size
    chunk = bytearray(size_mb * 1024 * 1024)
    # Fill it with data so it actually uses memory
    for i in range(0, len(chunk), 4096):
        chunk[i] = 1
    return chunk

def memory_leak_attack():
    print("=" * 50)
    print("MEMORY LEAK ATTACK - Gradual Increase")
    print("=" * 50)
    print(f"Duration: {DURATION} seconds")
    print(f"Chunk size: {CHUNK_SIZE_MB} MB every {INTERVAL} seconds")
    print(f"Max memory: {MAX_MEMORY_MB} MB")
    print("=" * 50)
    print("\nStarting attack... RAM will gradually increase.")
    print("Run monitor_anomaly.py in another terminal!")
    print("Press Ctrl+C to stop.\n")
    
    start_time = time.time()
    total_allocated = 0
    
    try:
        while time.time() - start_time < DURATION:
            elapsed = int(time.time() - start_time)
            
            if total_allocated < MAX_MEMORY_MB:
                # Allocate more memory
                chunk = allocate_memory(CHUNK_SIZE_MB)
                memory_chunks.append(chunk)
                total_allocated += CHUNK_SIZE_MB
                print(f"[{elapsed:3d}s] Allocated {CHUNK_SIZE_MB} MB (total: {total_allocated} MB)")
            else:
                print(f"[{elapsed:3d}s] Max memory reached, holding at {total_allocated} MB")
            
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        print("\n--- Attack stopped by user ---")
    finally:
        # Release memory
        memory_chunks.clear()
        print(f"Released {total_allocated} MB of memory")
        print("Attack finished.")

if __name__ == "__main__":
    memory_leak_attack()
