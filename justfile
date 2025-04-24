# Mallet-Sync Justfile (Windows/PowerShell, uv-based)
# Configure PowerShell as the shell with error handling
set shell := ["pwsh", "-NoLogo", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'"]

# Default recipe to run when just is called without arguments
default:
    @just --list

# Check if Python and other requirements are available
check-requirements:
    @Write-Host "Checking system requirements..." -ForegroundColor Cyan
    @$pythonExists = Get-Command python -ErrorAction SilentlyContinue
    @if (-not $pythonExists) { Write-Host "❌ Python not found. Please install Python 3.10 or newer." -ForegroundColor Red; exit 1 }
    @$pythonVersion = (python --version).Split(' ')[1]
    @Write-Host "✓ Found Python $pythonVersion" -ForegroundColor Green
    @$projectFileExists = Test-Path "pyproject.toml"
    @if (-not $projectFileExists) { Write-Host "❌ pyproject.toml not found. Are you in the correct directory?" -ForegroundColor Red; exit 1 }
    @Write-Host "✓ Project files verified" -ForegroundColor Green

# Install all dependencies (including dev) via uv and pyproject.toml
install: check-requirements
    @Write-Host "Setting up environment..." -ForegroundColor Cyan
    @$venvExists = Test-Path .venv
    @if ($venvExists) { Write-Host "Existing virtual environment found. Will be replaced." -ForegroundColor Yellow; Remove-Item -Recurse -Force .venv }
    @Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    @try { python -m venv .venv } catch { Write-Host "❌ Failed to create virtual environment: $_" -ForegroundColor Red; exit 1 }
    @Write-Host "✓ Virtual environment created" -ForegroundColor Green
    @Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    @try { .venv\Scripts\Activate.ps1 } catch { Write-Host "❌ Failed to activate virtual environment: $_" -ForegroundColor Red; exit 1 }
    @Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    @Write-Host "Installing uv..." -ForegroundColor Cyan
    @try { python -m pip install --upgrade pip; python -m pip install uv } catch { Write-Host "❌ Failed to install uv: $_" -ForegroundColor Red; exit 1 }
    @Write-Host "✓ uv installed" -ForegroundColor Green
    @Write-Host "Installing dependencies from pyproject.toml..." -ForegroundColor Cyan
    @try { uv sync --dev } catch { Write-Host "❌ Failed to sync dependencies: $_" -ForegroundColor Red; exit 1 }
    @Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
    @Write-Host "✅ Environment setup complete! Run 'just run' to start the application." -ForegroundColor Green

# Run the main application
run: check-venv
    @Write-Host "Starting mallet-sync application..." -ForegroundColor Cyan
    @try { uv run python src/mallet_sync/main.py "$@" } catch { Write-Host "❌ Application failed with error: $_" -ForegroundColor Red; exit 1 }

# Run the application excluding calibration files
run-test-only: check-venv
    @Write-Host "Starting mallet-sync with test audio only..." -ForegroundColor Cyan
    @uv run python src/mallet_sync/main.py --exclude-calibration

# Check if virtual environment exists and is activated
check-venv:
    @$venvExists = Test-Path .venv
    @if (-not $venvExists) { Write-Host "❌ Virtual environment not found. Please run 'just install' first." -ForegroundColor Red; exit 1 }

# Clean only output directory and Python caches
clean:
    @Write-Host "Cleaning output directories and caches..." -ForegroundColor Cyan
    @$outputExists = Test-Path output
    @if ($outputExists) { Remove-Item -Recurse -Force output; Write-Host "✓ Removed output directory" -ForegroundColor Green } else { Write-Host "✓ No output directory to clean" -ForegroundColor Green }
    @Write-Host "Cleaning Python cache files..." -ForegroundColor Cyan
    @Get-ChildItem -Recurse -Include __pycache__ -Directory | Remove-Item -Recurse -Force
    @Get-ChildItem -Recurse -Include *.pyc,*.pyo -File | Remove-Item -Force
    @Write-Host "✓ Python cache files removed" -ForegroundColor Green

# Clean everything: venv, output, build artifacts, caches
clean-all:
    @Write-Host "Performing complete cleanup..." -ForegroundColor Cyan
    @just clean
    @$venvExists = Test-Path .venv
    @if ($venvExists) { Remove-Item -Recurse -Force .venv; Write-Host "✓ Removed virtual environment" -ForegroundColor Green }
    @$distExists = Test-Path dist
    @if ($distExists) { Remove-Item -Recurse -Force dist; Write-Host "✓ Removed dist directory" -ForegroundColor Green }
    @$buildExists = Test-Path build
    @if ($buildExists) { Remove-Item -Recurse -Force build; Write-Host "✓ Removed build directory" -ForegroundColor Green }
    @foreach ($dir in Get-ChildItem -Directory -Filter "*.egg-info") { Remove-Item -Recurse -Force $dir; Write-Host "✓ Removed $($dir.Name)" -ForegroundColor Green }
    @Write-Host "✅ Complete cleanup finished" -ForegroundColor Green

# Freeze current environment to requirements.txt using uv
freeze: check-venv
    @Write-Host "Freezing current environment to requirements.txt..." -ForegroundColor Cyan
    @uv pip freeze > requirements.txt
    @Write-Host "✓ Environment requirements saved to requirements.txt" -ForegroundColor Green
