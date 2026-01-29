import socket
import threading
import time

TARGET_IP = "127.0.0.1"
TARGET_PORT = 9999  # Use a custom port for our own server
TOTAL_CONNECTIONS = 5000
HOLD_TIME = 600  # Keep connections open for 10 minutes

# Store all client sockets so they don't get garbage collected
all_sockets = []

def run_server():
    """A simple server that accepts and holds connections"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TARGET_IP, TARGET_PORT))
    server.listen(10000)  # Allow many pending connections
    print(f"[SERVER] Listening on port {TARGET_PORT}...")
    
    while True:
        try:
            client, addr = server.accept()
            all_sockets.append(client)  # Keep reference to prevent closing
        except:
            break

def connect_socket():
    """Create a client socket and connect to our server"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TARGET_IP, TARGET_PORT))
        all_sockets.append(s)  # Keep reference to prevent closing
        time.sleep(HOLD_TIME)
    except:
        pass

def start_attack():
    # Start the server first
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.5)  # Give server time to start
    
    print(f"--- ATTACK STARTED: Opening {TOTAL_CONNECTIONS} sockets ---")
    
    for i in range(TOTAL_CONNECTIONS):
        t = threading.Thread(target=connect_socket)
        t.daemon = True
        t.start()
        
        time.sleep(0.01)  # Small delay between connections
        
        if i % 100 == 0:
            print(f"Launched {i} sockets... (Total open: ~{len(all_sockets)})")
    
    print(f"\n[!] All sockets are OPEN. They will stay open for {HOLD_TIME} seconds.")
    print(f"Total sockets held: {len(all_sockets)}")
    print("Do not close this window until you are done collecting Anomaly data.")
    
    time.sleep(HOLD_TIME + 5)

if __name__ == "__main__":
    start_attack()