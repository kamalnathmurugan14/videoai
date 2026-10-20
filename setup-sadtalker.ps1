<#
  Installs SadTalker (realistic neural lip-sync, https://github.com/OpenTalker/SadTalker, Apache-2.0) for VideoAI on Windows.

  Run in PowerShell from the VideoAI project folder (the one with backend\ and frontend\):
      .\setup-sadtalker.ps1               # auto: NVIDIA GPU build if an NVIDIA card is found, otherwise CPU build
      .\setup-sadtalker.ps1 -Cpu          # force the CPU build
      .\setup-sadtalker.ps1 -Dir D:\SadTalker

  What it does: installs Python 3.10 if needed (SadTalker needs it), clones SadTalker next to this project, creates a
  separate virtualenv for it, installs PyTorch + pinned, known-good dependencies, downloads the model weights (~1.5 GB),
  runs a real self-test, and writes SADTALKER_* settings into backend\.env. Re-running is safe (it skips finished steps).
#>
param(
  [string]$Dir = (Join-Path (Split-Path -Parent $PSScriptRoot) "SadTalker"),
  [switch]$Cpu
)
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$argv) {
  & $exe @argv
  if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $exe $($argv -join ' ')" }
}

# ---------------------------------------------------------------- prerequisites
Step "Checking git and FFmpeg"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is not installed. Run: winget install Git.Git   then open a new PowerShell." }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "FFmpeg is not on PATH. Run: winget install Gyan.FFmpeg   then open a new PowerShell." }

Step "Finding Python 3.10 (SadTalker does not work on newer Python)"
function Find-Py310 {
  try { $p = (& py -3.10 -c "import sys; print(sys.executable)" 2>$null); if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
  $c = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
  if (Test-Path $c) { return $c }
  return $null
}
$py310 = Find-Py310
if (-not $py310) {
  Write-Host "Installing Python 3.10 with winget..."
  Run winget @("install", "-e", "--id", "Python.Python.3.10", "--accept-package-agreements", "--accept-source-agreements")
  $py310 = Find-Py310
  if (-not $py310) { throw "Python 3.10 was installed but not found. Open a new PowerShell and run this script again." }
}
Write-Host "Using $py310"

# ---------------------------------------------------------------- code
Step "Getting SadTalker into $Dir"
if (-not (Test-Path (Join-Path $Dir "inference.py"))) {
  Run git @("clone", "--depth", "1", "https://github.com/OpenTalker/SadTalker.git", $Dir)
} else { Write-Host "Already present." }

# ---------------------------------------------------------------- python env
$venvPy = Join-Path $Dir "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Step "Creating SadTalker's own virtualenv"
  Run $py310 @("-m", "venv", (Join-Path $Dir "venv"))
}
Run $venvPy @("-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<70")

$gpu = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  & nvidia-smi *> $null
  if ($LASTEXITCODE -eq 0) { $gpu = $true }
}
Step ("Installing PyTorch (" + ($(if ($gpu) { "NVIDIA GPU / CUDA 11.8" } else { "CPU only - rendering will be slow" })) + ")")
$torch = @("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2")
if ($gpu) { Run $venvPy (@("-m", "pip", "install") + $torch + @("--index-url", "https://download.pytorch.org/whl/cu118")) }
else { Run $venvPy (@("-m", "pip", "install") + $torch) }

Step "Installing SadTalker dependencies (pinned versions known to work together)"
Run $venvPy @("-m", "pip", "install",
  "numpy==1.23.4", "face_alignment==1.3.5", "imageio==2.19.3", "imageio-ffmpeg==0.4.7", "librosa==0.9.2", "numba==0.58.1",
  "resampy==0.3.1", "pydub==0.25.1", "scipy==1.10.1", "kornia==0.6.8", "tqdm", "yacs==0.1.8", "pyyaml", "joblib==1.1.0",
  "scikit-image==0.19.3", "opencv-python==4.7.0.72", "basicsr==1.4.2", "facexlib==0.3.0", "gfpgan==1.3.8", "av", "safetensors")

# ---------------------------------------------------------------- weights
Step "Downloading model weights (about 1.5 GB, resumable)"
$rel = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
$files = @(
  @("checkpoints\mapping_00109-model.pth.tar", "$rel/mapping_00109-model.pth.tar"),
  @("checkpoints\mapping_00229-model.pth.tar", "$rel/mapping_00229-model.pth.tar"),
  @("checkpoints\SadTalker_V0.0.2_256.safetensors", "$rel/SadTalker_V0.0.2_256.safetensors"),
  @("checkpoints\SadTalker_V0.0.2_512.safetensors", "$rel/SadTalker_V0.0.2_512.safetensors"),
  @("gfpgan\weights\alignment_WFLW_4HG.pth", "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth"),
  @("gfpgan\weights\detection_Resnet50_Final.pth", "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
  @("gfpgan\weights\GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"),
  @("gfpgan\weights\parsing_parsenet.pth", "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth")
)
foreach ($f in $files) {
  $dest = Join-Path $Dir $f[0]
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 1000000)) { Write-Host "ok  $($f[0])"; continue }
  Write-Host "get $($f[0])"
  Run curl.exe @("-L", "--fail", "--retry", "3", "-C", "-", "-o", $dest, $f[1])
}

# ---------------------------------------------------------------- self-test
Step "Self-test: animating a SadTalker example image (this proves it works on this PC)"
$img = Get-ChildItem (Join-Path $Dir "examples\source_image") -Filter *.png | Select-Object -First 1
$wav = Get-ChildItem (Join-Path $Dir "examples\driven_audio") -Filter *.wav | Select-Object -First 1
$short = Join-Path $Dir "selftest_audio.wav"
Run $venvPy @("-c", "import wave,sys; r=wave.open(sys.argv[1]); w=wave.open(sys.argv[2],'wb'); w.setparams(r.getparams()); w.writeframes(r.readframes(int(r.getframerate()*3))); w.close()", $wav.FullName, $short)
$out = Join-Path $Dir "results_selftest"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
$sw = [Diagnostics.Stopwatch]::StartNew()
Push-Location $Dir
try { Run $venvPy @("inference.py", "--driven_audio", $short, "--source_image", $img.FullName, "--result_dir", $out, "--still", "--preprocess", "full") }
finally { Pop-Location }
$sw.Stop()
$video = Get-ChildItem $out -Filter *.mp4 | Select-Object -First 1
if (-not $video) { throw "Self-test produced no video - see the messages above." }
Write-Host ("Self-test OK: 3 s of video in {0:N0} s -> {1}" -f $sw.Elapsed.TotalSeconds, $video.FullName) -ForegroundColor Green

# ---------------------------------------------------------------- configure VideoAI
Step "Writing SadTalker settings to backend\.env"
$envFile = Join-Path $PSScriptRoot "backend\.env"
if (-not (Test-Path $envFile)) { Copy-Item (Join-Path $PSScriptRoot ".env.example") $envFile }
$settings = [ordered]@{
  "SADTALKER_DIR" = (Resolve-Path $Dir).Path
  "SADTALKER_PYTHON" = (Resolve-Path $venvPy).Path
  "SADTALKER_ENHANCER" = $(if ($gpu) { "gfpgan" } else { "" })   # face enhancer: sharper, but too slow on CPU
  "SADTALKER_SIZE" = "256"
}
$lines = Get-Content $envFile | Where-Object { $_ -notmatch "^(SADTALKER_DIR|SADTALKER_PYTHON|SADTALKER_ENHANCER|SADTALKER_SIZE)=" }
foreach ($k in $settings.Keys) { $lines += "$k=$($settings[$k])" }
[IO.File]::WriteAllLines($envFile, [string[]]$lines)  # UTF-8 without BOM (a BOM would corrupt the first setting)

Write-Host "`nDone. Restart the VideoAI backend, open http://localhost:5173/system and press 'Test SadTalker'." -ForegroundColor Green
Write-Host "Then in the editor: Look & feel -> Animation engine -> Realistic (SadTalker)." -ForegroundColor Green
if (-not $gpu) { Write-Host "No NVIDIA GPU was used: expect roughly 10+ minutes per 10 seconds of video." -ForegroundColor Yellow }
