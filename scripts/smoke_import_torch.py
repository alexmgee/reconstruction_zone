"""
Minimal torch import smoke test for Nuitka packaging.

Compile this standalone to verify the singledispatch YAML fix works
before spending 30 minutes on a full GUI build.

Usage (PowerShell):
    & 'C:\Python314\python.exe' -m nuitka `
      --standalone --assume-yes-for-downloads `
      --include-package=torch --include-package=torchvision `
      --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
      --module-parameter=torch-disable-jit=yes `
      --show-anti-bloat-changes `
      --show-source-changes=torch.library `
      --show-source-changes=torch._refs `
      --report=dist_test/nuitka-report.xml `
      --output-dir=dist_test `
      scripts/smoke_import_torch.py

    & .\dist_test\smoke_import_torch.dist\smoke_import_torch.exe
"""
import torch
import torch._refs
import torchvision

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
print("OK")
