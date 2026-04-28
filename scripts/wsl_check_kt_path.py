"""Print kt_kernel install path and explore its modules."""
import os
import sys

import kt_kernel  # noqa: E402

print("kt_kernel path:", kt_kernel.__path__)
print("kt_kernel file:", getattr(kt_kernel, "__file__", "N/A"))
print()
# Check for key submodules
for mod in ["experts", "experts_base", "utils", "generate_gpu_experts_masks"]:
    try:
        m = getattr(kt_kernel, mod, None)
        if m is not None:
            print(f"  {mod}: {m}")
    except Exception as e:
        print(f"  {mod}: ERROR {e}")
print()

# Also try to find experts_base source
for p in sys.path:
    candidate = f"{p}/kt_kernel"
    if os.path.isdir(candidate):
        print(f"Found kt_kernel dir: {candidate}")
        for root, dirs, files in os.walk(candidate):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    print(f"  {full}")