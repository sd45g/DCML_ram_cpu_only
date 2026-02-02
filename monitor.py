import psutil
import time

def get_size(bytes):
    """Convert bytes to MB"""
    return bytes / (1024 * 1024)

class SystemMonitor:
    def __init__(self):
        self.prev_net = psutil.net_io_counters()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_time = time.time()
        # Prime CPU counter so first call isn't 0.0
        psutil.cpu_percent(interval=None)
        
    def get_metrics(self):
        # 1. CPU & RAM
        # interval=None means non-blocking (instant look)
        cpu = psutil.cpu_percent(interval=None) 
        ram = psutil.virtual_memory().percent
        
        # 2. Process Count
        try:
            process_count = len(psutil.pids())
        except:
            process_count = 0
        
        # 3. Disk & Network Rates
        curr_net = psutil.net_io_counters()
        curr_disk = psutil.disk_io_counters()
        curr_time = time.time()
        
        # Avoid division by zero
        dt = curr_time - self.prev_time
        if dt == 0: dt = 0.001 
        
        # Calculate Rates (MB/s)
        # Network Sent (Key for exfiltration)
        sent_bytes = curr_net.bytes_sent - self.prev_net.bytes_sent
        net_sent_speed = get_size(sent_bytes) / dt
        
        # Disk Read/Write
        read_bytes = curr_disk.read_bytes - self.prev_disk.read_bytes
        write_bytes = curr_disk.write_bytes - self.prev_disk.write_bytes
        disk_read_speed = get_size(read_bytes) / dt
        disk_write_speed = get_size(write_bytes) / dt
        
        # Update previous values
        self.prev_net = curr_net
        self.prev_disk = curr_disk
        self.prev_time = curr_time
        
        return {
            "cpu": cpu,
            "ram": ram,
            "processes": process_count,
            "disk_read": disk_read_speed,
            "disk_write": disk_write_speed,
            "net_sent": net_sent_speed
        }

if __name__ == "__main__":
    monitor = SystemMonitor()
    print("Monitor initialized. Run this file to test...")
    
    try:
        # Initial sleep to establish baseline
        time.sleep(1)
        while True:
            metrics = monitor.get_metrics()
            print(f"CPU: {metrics['cpu']:5.1f}% | RAM: {metrics['ram']:5.1f}% | "
                  f"Disk R: {metrics['disk_read']:5.1f} MB/s | "
                  f"Net Sent: {metrics['net_sent']:5.1f} MB/s")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
