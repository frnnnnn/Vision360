import socket
import threading
from typing import List, Dict

class ScannerService:
    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def scan_network(self) -> List[Dict[str, str]]:
        local_ip = self.get_local_ip()
        subnet = ".".join(local_ip.split(".")[:3])
        found_devices = []
        threads = []

        # Ports to check: 80 (HTTP), 8080 (Alt HTTP), 554 (RTSP), 8081 (Alt)
        target_ports = [80, 8080, 554, 8081]

        def check_ip(ip):
            for port in target_ports:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5) # Fast timeout
                result = s.connect_ex((ip, port))
                if result == 0:
                    try:
                        hostname = socket.gethostbyaddr(ip)[0]
                    except:
                        hostname = "Unknown"
                    
                    found_devices.append({
                        "ip": ip,
                        "port": str(port),
                        "hostname": hostname,
                        "url": f"http://{ip}:{port}" if port != 554 else f"rtsp://{ip}:{port}/live"
                    })
                    s.close()
                    break # Found one open port, assume device is active
                s.close()

        # Scan range 1-254
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            if ip == local_ip: continue # Skip self
            
            t = threading.Thread(target=check_ip, args=(ip,))
            threads.append(t)
            t.start()
            
            # Limit concurrency slightly to avoid OS limits
            if len(threads) > 50:
                for t in threads: t.join()
                threads = []

        for t in threads: t.join()
        
        return found_devices

scanner_service = ScannerService()
