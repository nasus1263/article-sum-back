# run.ps1
$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " FastAPI Service Launcher & Environment Builder " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Check for .env file
if (-not (Test-Path -Path ".env")) {
    if (Test-Path -Path ".env.example") {
        Write-Host "[+] Creating .env file from .env.example..." -ForegroundColor Yellow
        Copy-Item -Path ".env.example" -Destination ".env"
        Write-Host "[!] Please configure your API keys in the '.env' file." -ForegroundColor Magenta
    } else {
        Write-Host "[-] .env.example not found. Skipping .env creation." -ForegroundColor Yellow
    }
}

# 2. Check for Python installation
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "[+] Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[Error] Python was not found in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python and make sure it is added to your PATH environment variables." -ForegroundColor Yellow
    Exit 1
}

# 3. Create Virtual Environment if not exists
$venvDir = ".venv"
if (-not (Test-Path -Path $venvDir)) {
    Write-Host "[+] Creating virtual environment in $venvDir..." -ForegroundColor Yellow
    try {
        & python -m venv $venvDir
        Write-Host "[+] Virtual environment created successfully." -ForegroundColor Green
    } catch {
        Write-Host "[Error] Failed to create virtual environment." -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[+] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 4. Install requirements
if (Test-Path -Path "requirements.txt") {
    Write-Host "[+] Installing/Updating dependencies from requirements.txt..." -ForegroundColor Yellow
    try {
        & .\.venv\Scripts\pip.exe install -r requirements.txt
        Write-Host "[+] Dependencies installation complete." -ForegroundColor Green
    } catch {
        Write-Host "[Error] Failed to install dependencies." -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[-] requirements.txt not found. Skipping package installation." -ForegroundColor Yellow
}

# 5. Run FastAPI App
Write-Host "[+] Starting Uvicorn server on port 9000 (accessible from local network)..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host "-----------------------------------------" -ForegroundColor Cyan

try {
    & .\.venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 9000
} catch {
    Write-Host "[Error] Failed to run uvicorn." -ForegroundColor Red
    Write-Host "Please check if uvicorn is installed correctly inside the virtual environment." -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Red
    Exit 1
}

