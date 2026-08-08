import subprocess
import webbrowser
import time
import sys
import os
import socket

def wait_for_port(port, host='localhost', timeout=30):
    """Wait until a port starts accepting TCP connections."""
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
            if time.time() - start_time > timeout:
                return False

def check_and_install_requirements(base_dir, python_exec):
    """Install requirements if venv is freshly created or missing packages."""
    req_file = os.path.join(base_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Checking dependencies...")
        try:
            subprocess.check_call([python_exec, "-m", "pip", "install", "-r", req_file, "--quiet"])
            print("Dependencies verified.")
        except Exception as e:
            print(f"Warning: Dependency check failed: {e}")

def kill_stale_processes(port=8001):
    """Gracefully kill any process occupying the target port."""
    print(f"Checking for stale processes on port {port}...")
    try:
        # Cross-platform approach (lsof for unix)
        result = subprocess.run(["lsof", f"-ti:{port}"], capture_output=True, text=True)
        pids = result.stdout.strip()
        if pids:
            for pid in pids.split("\n"):
                if pid:
                    os.kill(int(pid), 9)
            print(f"  Cleared stale process(es) on port {port}.")
            time.sleep(1)
        else:
            print(f"  Port {port} is free.")
    except Exception:
        pass

def main():
    print("==================================================")
    print("      TURBOJET DIGITAL TWIN LAUNCHER")
    print("==================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Resolve Python Interpreter
    venv_python = os.path.join(base_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        python_exec = venv_python
        print("✓ Using virtual environment Python.")
    else:
        python_exec = sys.executable
        print("⚠ Venv not found. Using system Python.")
        print("  Recommendation: run `python -m venv venv` for a clean environment.")

    # 2. Check Dependencies
    check_and_install_requirements(base_dir, python_exec)

    # 3. Clean Port 8001
    kill_stale_processes(8001)
    
    # 4. Start the Server
    server_script = os.path.join(base_dir, "server.py")
    print("\nStarting the FastAPI backend server...")
    
    process = subprocess.Popen([python_exec, server_script])
    
    # 5. Wait for Server to Initialize ML Models
    print("Waiting for the server to load ML models (this may take a few seconds)...")
    if wait_for_port(8001, timeout=60):
        url = "http://localhost:8001"
        print(f"\n✓ Server is up! Opening {url} in your default browser...")
        time.sleep(1) # Final micro-sleep to ensure static routes are mounted
        webbrowser.open(url)
    else:
        print("\n❌ Server failed to bind to port 8001 within the timeout.")
        process.terminate()
        sys.exit(1)
    
    print("\n[ACTIVE] Keep this script running to interact with the dashboard.")
    print("Press Ctrl+C to stop the server and exit.")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping server gracefully...")
        process.terminate()
        process.wait()
        print("Server stopped. Goodbye!")

if __name__ == "__main__":
    main()
