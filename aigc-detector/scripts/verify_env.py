"""Verify the PyTorch install can actually execute kernels on this GPU.

Run this FIRST, before any model code. On Blackwell cards (RTX 50-series,
compute capability sm_120) a PyTorch wheel built against older CUDA will
report `torch.cuda.is_available() == True`, load models happily, and then
die on the first real forward pass with:

    CUDA error: no kernel image is available for execution on the device

The tensor-allocation check is not sufficient to catch this -- only actually
running a kernel is. That is what the matmul below does.

If this script fails on arch support:

    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Usage:
    python scripts/verify_env.py
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed.")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    print(f"torch version      : {torch.__version__}")
    print(f"built for CUDA     : {torch.version.cuda}")
    print(f"cuda available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nFAIL: no CUDA device visible. Check the NVIDIA driver "
              "(`nvidia-smi`) and that this is a CUDA build of torch.")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"device             : {name}")
    print(f"compute capability : sm_{major}{minor}")
    print(f"total VRAM         : {total_gb:.1f} GB")
    print(f"compiled archs     : {' '.join(arch_list)}")

    wanted = f"sm_{major}{minor}"
    if wanted not in arch_list:
        print(f"\nFAIL: this torch build has no kernels for {wanted}.")
        print("  Blackwell (sm_120) needs a CUDA 12.8+ build:")
        print("  pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    # The real test: allocate AND execute. Allocation alone can succeed on a
    # build that cannot run kernels, which is exactly the trap.
    try:
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        y = (x @ x).float().sum().item()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - we want to surface anything
        print(f"\nFAIL: kernel execution failed: {exc}")
        return 1

    if y != y:  # NaN check
        print("\nWARN: matmul produced NaN. Numerically odd but kernels run.")

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"fp16 matmul        : OK")
    print(f"free VRAM after    : {free_gb:.1f} GB")

    if total_gb < 10:
        print("\nNOTE: <10GB VRAM detected. Use the frozen-backbone plan "
              "(feature caching); do not attempt full fine-tuning.")

    print("\nPASS: environment is usable. Pin this torch version in "
          "requirements.txt now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
