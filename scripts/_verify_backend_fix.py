"""Quick verification that Blackwell GPU returns pytorch_cuda backend."""
import sys
sys.path.insert(0, "/mnt/c/Users/Qchau/Documents/GitHub/Astra")
from astra.inference.heterogeneous import _detect_backend, _kt_backend, _kt

backend, mod = _detect_backend()
print(f"backend={backend}")
print(f"mod type={type(mod).__name__}")
print(f"_kt_backend={_kt_backend}")
print(f"_kt is None={_kt is None}")
assert backend == "pytorch_cuda", f"Expected pytorch_cuda, got {backend}"
assert _kt_backend == "pytorch_cuda", f"Module-level _kt_backend mismatch: {_kt_backend}"
print("PASS: Blackwell GPU correctly detected as pytorch_cuda")
