# latency_test.py
import socket, time, statistics, os

DST_IP   = "192.168.30.1"
DST_PORT = 2090
IFACE    = "enp65s0f1np1"
COUNT    = 1000
SIZE     = 64

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, IFACE.encode())
s.bind(("192.168.40.1", 0))
s.settimeout(0.5)

latencies = []
msg = b"X" * SIZE

for i in range(COUNT):
    t0 = time.perf_counter_ns()
    s.sendto(msg, (DST_IP, DST_PORT))
    try:
        s.recvfrom(4096)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # µs
    except socket.timeout:
        print(f"[{i}] timeout")

s.close()

if latencies:
    print(f"\nRTT over {len(latencies)} packets:")
    print(f"  Min : {min(latencies):.2f} µs")
    print(f"  Max : {max(latencies):.2f} µs")
    print(f"  Avg : {statistics.mean(latencies):.2f} µs")
    print(f"  p50 : {statistics.median(latencies):.2f} µs")