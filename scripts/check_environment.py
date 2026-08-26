"""Report the supported runtime versions and CUDA capability without failing on CPU."""
from __future__ import annotations

import platform
import sys


def version(package: str) -> str:
    try:
        from importlib.metadata import version as installed_version

        return installed_version(package)
    except Exception:
        return "not installed"


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {version('torch')}")
    print(f"Transformers: {version('transformers')}")
    print(f"PEFT: {version('peft')}")
    print(f"Datasets: {version('datasets')}")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("WARNING: this project supports Python 3.11 or 3.12; do not install dependencies here.")
        return 2

    try:
        import torch
    except ImportError:
        print("\nCUDA available: unavailable (PyTorch is not installed)")
        return 1

    available = torch.cuda.is_available()
    print(f"\nCUDA available: {available}")
    print(f"CUDA version (PyTorch): {torch.version.cuda or 'CPU build'}")
    if available:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        allocated = torch.cuda.memory_allocated(index) / 1024**3
        total = properties.total_memory / 1024**3
        print(f"GPU: {properties.name}")
        print(f"VRAM: {total:.2f} GiB total; {total - allocated:.2f} GiB available to PyTorch")
    else:
        print("GPU: not available to PyTorch")
        print("VRAM: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

