from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"python_version={sys.version.split()[0]}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("error=CUDA is not available in this environment.")
        return 1

    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}={torch.cuda.get_device_name(index)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
