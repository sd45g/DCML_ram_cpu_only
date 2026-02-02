import random
import socket
import time
import os

def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting Network Flood (Data Exfiltration Simulation)...")
    
    # Dummy target (won't actually hurt anyone, just generates outbound traffic)
    target_ip = "127.0.0.1" 
    target_port = 9999
    
    # Create a large packet (60KB - Safe for Windows UDP)
    packet = os.urandom(60 * 1024)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    sent_mb = 0
    start_time = time.time()
    
    try:
        while True:
            # TUNED UPDATE: Variable bursts (5-40 packets)
            # Reason: Creates unpredictable traffic spikes (Harder to predict, more realistic)
            num_packets = random.randint(5, 40)
            
            for _ in range(num_packets):
                sock.sendto(packet, (target_ip, target_port))
                sent_mb += 64 / 1024
                
            # Random Sleep: 0.05s - 0.2s
            # Reason: Simulates human/network variability vs machine regularity
            time.sleep(random.uniform(0.05, 0.2))
            
            # Print status
            elapsed = time.time() - start_time
            if elapsed > 1:
                print(f"\rSent: {sent_mb:.2f} MB", end="")
            
    except KeyboardInterrupt:
        print("\nStopping...")
        sock.close()

if __name__ == "__main__":
    main()
