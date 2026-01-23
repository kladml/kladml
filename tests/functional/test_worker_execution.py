import pytest
import subprocess
import shutil
import sys
from pathlib import Path

# Detect Runtime (prefer Docker for GPU support)
def get_container_runtime():
    if shutil.which("docker"):
        return "docker"
    if shutil.which("podman"):
        return "podman"
    return None

RUNTIME = get_container_runtime()

# Path to the verification script relative to project root
VERIFY_SCRIPT = "sdk/src/kladml/cli/verify_device_job.py"

def run_kladml_cli(device: str):
    """Run the kladml run local command."""
    if not RUNTIME:
        pytest.fail("No container runtime found")
        
    cmd = [
        "kladml", "run", "local",
        str(VERIFY_SCRIPT),
        "--device", device,
        "--runtime", RUNTIME
    ]
    # Run from project root so relative path works
    cwd = Path(__file__).parents[3] 
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

@pytest.mark.skipif(not RUNTIME, reason="No container runtime (docker/podman) found")
class TestWorkerExecution:
    
    @pytest.fixture(scope="class", autouse=True)
    def ensure_images_built(self):
        """Ensure base worker images are built before running tests."""
        # Check if CPU image exists
        res = subprocess.run([RUNTIME, "images", "-q", "ghcr.io/kladml/worker:cpu"], capture_output=True)
        if not res.stdout:
            pytest.skip(f"Worker images not built in {RUNTIME}. Run ./scripts/build_workers.sh first.")

    def test_cpu_worker(self):
        """Test standard CPU execution."""
        result = run_kladml_cli("cpu")
        print(f"--- CMD: {' '.join(result.args)} ---")
        print(f"--- STDOUT ---\n{result.stdout}")
        print(f"--- STDERR ---\n{result.stderr}")
        
        assert result.returncode == 0
        # The verification script prints "Actual: cpu" or similar
        assert "Device: cpu" in result.stdout or "Actual: cpu" in result.stdout
        assert "FULL SUCCESS" in result.stdout

    # GPU Detection
    try:
        has_cuda = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except (FileNotFoundError, OSError):
        has_cuda = False
        
    import platform
    has_mps = platform.system() == "Darwin" and platform.processor() == "arm"
    
    @pytest.mark.skipif(not (has_cuda or has_mps), reason="No GPU (CUDA or MPS) detected")
    def test_cuda_worker(self):
        """Test CUDA execution (only if GPU present)."""
        # Ensure CUDA image exists
        res = subprocess.run([RUNTIME, "images", "-q", "ghcr.io/kladml/worker:cuda12"], capture_output=True)
        if not res.stdout:
            pytest.skip("CUDA worker image not built.")
            
        result = run_kladml_cli("cuda")
        print(result.stdout)
        print(result.stderr)
        
        assert result.returncode == 0
        assert "Actual: cuda" in result.stdout
        assert "FULL SUCCESS" in result.stdout

    def test_mps_worker(self):
        """Test MPS execution request."""
        # This test ensures that requesting 'mps' doesn't crash, 
        # even if it typically falls back to CPU on Linux/Docker.
        result = run_kladml_cli("mps")
        print(result.stdout)
        print(result.stderr)
        
        assert result.returncode == 0
        
        if sys.platform == "darwin":
             # On native mac, verifying native runner might yield MPS
             # But here we force --runtime docker, so we expect CPU fallback
             # unless we are on a very specific setup.
             # The verify script handles the warning.
             assert "Actual: cpu" in result.stdout or "Actual: mps" in result.stdout
        else:
             # On Linux, requesting MPS should fallback to CPU image and warn
             assert "Actual: cpu" in result.stdout
             assert "Running MPS request on CPU" in result.stdout or "FULL SUCCESS" in result.stdout
