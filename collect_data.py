"""
DCML Project - Automated Data Collection
Collects 1000 samples (500 normal + 500 anomaly) for training.
"""

import psutil
import time
import csv
import os
import multiprocessing
import threading
import sys

DATA_FILE = "my_system_data.csv"
BACKUP_FILE = "my_system_data_backup.csv"

# Configuration
NORMAL_SAMPLES = 500
ANOMALY_SAMPLES_PER_TYPE = 170  # ~500 total across 3 attack types


def get_system_status(label):
    """Collect current CPU and RAM usage"""
    return {
        "timestamp": time.time(),
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent,
        "label": label
    }


def cpu_stress_worker(stop_event):
    """Worker that stresses CPU"""
    while not stop_event.is_set():
        _ = sum(i * i for i in range(10000))


def memory_stress_worker(stop_event, size_mb=50):
    """Worker that allocates memory"""
    data = []
    while not stop_event.is_set():
        try:
            # Allocate chunks
            chunk = bytearray(size_mb * 1024 * 1024)
            data.append(chunk)
            time.sleep(1)
        except MemoryError:
            break
    # Release memory
    data.clear()


def collect_samples(writer, f, label, count, description):
    """Collect n samples with given label"""
    print(f"\n{'='*50}")
    print(f"Collecting {count} {description} samples...")
    print(f"{'='*50}")
    
    for i in range(count):
        stats = get_system_status(label)
        writer.writerow(stats)
        f.flush()
        
        progress = (i + 1) / count * 100
        print(f"\r[{progress:5.1f}%] Sample {i+1}/{count} - CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}%", end="")
        time.sleep(1)
    
    print(f"\n✓ Completed {count} {description} samples")


def collect_with_cpu_attack(writer, f, count):
    """Collect anomaly samples during CPU attack"""
    print(f"\n{'='*50}")
    print(f"Collecting {count} ANOMALY samples (CPU Attack)...")
    print(f"{'='*50}")
    
    stop_event = threading.Event()
    
    # Start CPU stress workers (gradual increase)
    workers = []
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    for _ in range(num_workers):
        worker = threading.Thread(target=cpu_stress_worker, args=(stop_event,))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    print(f"Started {num_workers} CPU stress workers...")
    
    try:
        for i in range(count):
            stats = get_system_status(1)  # Anomaly label
            writer.writerow(stats)
            f.flush()
            
            progress = (i + 1) / count * 100
            print(f"\r[{progress:5.1f}%] Sample {i+1}/{count} - CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}%", end="")
            time.sleep(1)
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=1)
    
    print(f"\n✓ Completed {count} CPU attack samples")


def collect_with_memory_attack(writer, f, count):
    """Collect anomaly samples during memory attack"""
    print(f"\n{'='*50}")
    print(f"Collecting {count} ANOMALY samples (Memory Attack)...")
    print(f"{'='*50}")
    
    stop_event = threading.Event()
    
    # Start memory stress worker
    worker = threading.Thread(target=memory_stress_worker, args=(stop_event, 30))
    worker.daemon = True
    worker.start()
    
    print("Started memory stress worker...")
    
    try:
        for i in range(count):
            stats = get_system_status(1)  # Anomaly label
            writer.writerow(stats)
            f.flush()
            
            progress = (i + 1) / count * 100
            print(f"\r[{progress:5.1f}%] Sample {i+1}/{count} - CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}%", end="")
            time.sleep(1)
    finally:
        stop_event.set()
        worker.join(timeout=2)
    
    print(f"\n✓ Completed {count} memory attack samples")


def collect_with_mixed_attack(writer, f, count):
    """Collect anomaly samples during mixed CPU+memory attack"""
    print(f"\n{'='*50}")
    print(f"Collecting {count} ANOMALY samples (Mixed Attack)...")
    print(f"{'='*50}")
    
    stop_event = threading.Event()
    workers = []
    
    # Start 2 CPU workers
    for _ in range(2):
        worker = threading.Thread(target=cpu_stress_worker, args=(stop_event,))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    # Start 1 memory worker
    mem_worker = threading.Thread(target=memory_stress_worker, args=(stop_event, 20))
    mem_worker.daemon = True
    mem_worker.start()
    workers.append(mem_worker)
    
    print("Started mixed stress workers...")
    
    try:
        for i in range(count):
            stats = get_system_status(1)  # Anomaly label
            writer.writerow(stats)
            f.flush()
            
            progress = (i + 1) / count * 100
            print(f"\r[{progress:5.1f}%] Sample {i+1}/{count} - CPU: {stats['cpu']:5.1f}% | RAM: {stats['ram']:5.1f}%", end="")
            time.sleep(1)
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=2)
    
    print(f"\n✓ Completed {count} mixed attack samples")


def main():
    print("\n" + "#" * 60)
    print("#  DCML Automated Data Collection")
    print("#  Target: 1000 samples (500 normal + 500 anomaly)")
    print("#" * 60)
    
    # Backup existing data
    if os.path.exists(DATA_FILE):
        import shutil
        shutil.copy(DATA_FILE, BACKUP_FILE)
        print(f"\n✓ Backed up existing data to: {BACKUP_FILE}")
    
    # Remove old file to start fresh
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    
    # Initialize psutil CPU monitoring
    psutil.cpu_percent(interval=None)
    
    total_start = time.time()
    
    with open(DATA_FILE, "w", newline="") as f:
        fieldnames = ["timestamp", "cpu", "ram", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 1. Collect normal samples
        collect_samples(writer, f, 0, NORMAL_SAMPLES, "NORMAL")
        
        # 2. Collect anomaly samples with CPU attack
        collect_with_cpu_attack(writer, f, ANOMALY_SAMPLES_PER_TYPE)
        
        # 3. Collect anomaly samples with memory attack
        collect_with_memory_attack(writer, f, ANOMALY_SAMPLES_PER_TYPE)
        
        # 4. Collect anomaly samples with mixed attack
        collect_with_mixed_attack(writer, f, ANOMALY_SAMPLES_PER_TYPE)
    
    total_time = time.time() - total_start
    total_samples = NORMAL_SAMPLES + (ANOMALY_SAMPLES_PER_TYPE * 3)
    
    print("\n" + "#" * 60)
    print("#  DATA COLLECTION COMPLETE!")
    print("#" * 60)
    print(f"\n✓ Total samples collected: {total_samples}")
    print(f"✓ Normal samples: {NORMAL_SAMPLES}")
    print(f"✓ Anomaly samples: {ANOMALY_SAMPLES_PER_TYPE * 3}")
    print(f"✓ Time taken: {total_time/60:.1f} minutes")
    print(f"\n✓ Data saved to: {DATA_FILE}")
    print("\nNext steps:")
    print("  1. Run: python train_model.py")
    print("  2. Run: python analyze_model.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n--- Collection stopped by user ---")
        sys.exit(1)
