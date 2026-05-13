param(
    [Parameter(Mandatory = $true)]
    [string]$DistDir
)

$ErrorActionPreference = "Stop"

$dist = Resolve-Path -LiteralPath $DistDir
$cv2Dir = Join-Path $dist "cv2"

if (-not (Test-Path -LiteralPath $cv2Dir)) {
    throw "cv2 directory not found in bundle: $cv2Dir"
}

$sources = @(
    "C:\Windows\System32\nvcuvid.dll",
    "C:\Windows\System32\nvEncodeAPI64.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppitc64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppig64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cufft64_12.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppif64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppim64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppial64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppist64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppidei64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppicc64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cublas64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cublasLt64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudart64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nppc64_13.dll",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudnn64_9.dll"
)

$copied = 0
foreach ($src in $sources) {
    if (-not (Test-Path -LiteralPath $src)) {
        throw "Required OpenCV dependency not found: $src"
    }
    Copy-Item -LiteralPath $src -Destination $cv2Dir -Force
    Write-Host "copied $src"
    $copied += 1
}

Write-Host "Copied $copied OpenCV CUDA/NVIDIA DLLs to $cv2Dir"
