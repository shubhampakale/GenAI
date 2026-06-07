import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

import time
time.sleep(3)  # Give it time to start or crash

# Check if it's still running
if process.poll() is not None:
    print("❌ Server CRASHED immediately!")
    print("STDOUT:", process.stdout.read().decode())
    print("STDERR:", process.stderr.read().decode())
else:
    print("✅ Server is running (PID:", process.pid, ")")
    process.terminate()