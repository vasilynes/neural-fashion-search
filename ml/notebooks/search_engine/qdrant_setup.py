# This is a module to setup qdrant for Colab
import subprocess
import time
from pathlib import Path

def download_and_setup_qdrant():
    """Download and extract Qdrant binary"""
    subprocess.run([
        'wget', '-q', '-O', 'qdrant.tar.gz',
        'https://github.com/qdrant/qdrant/releases/download/v1.17.1/qdrant-x86_64-unknown-linux-musl.tar.gz'
    ], check=True)
    
    subprocess.run(['tar', '-xzf', 'qdrant.tar.gz'], check=True)
    
    subprocess.run(['chmod', '+x', 'qdrant'], check=True)
    
    print('Qdrant binary ready')

def start_qdrant():
    """Start Qdrant server"""
    proc = subprocess.Popen(
        ['./qdrant'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)  # Wait for startup
    print('Qdrant started!')
    return proc

def prepare_snapshot(snapshot_source, snapshot_dest):
    """Copy snapshot to expected location"""
    Path(snapshot_dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', snapshot_source, snapshot_dest], check=True)
    print(f"Snapshot copied to {snapshot_dest}")

def setup(snapshot_source, snapshot_dest):
    download_and_setup_qdrant()
    proc = start_qdrant()
    prepare_snapshot(snapshot_source, snapshot_dest)
    return proc

