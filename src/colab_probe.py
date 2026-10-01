import os
import platform
import subprocess

print("python", platform.python_version())
print("cwd", os.getcwd())
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch_error", repr(exc))

try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip())
except Exception as exc:
    print("nvidia_smi_error", repr(exc))
