# justfile for mallet-sync (Windows/PowerShell, uv-based)
# Configure PowerShell as the shell
set shell := ["pwsh", "-NoLogo", "-Command"]

# List available recipes with descriptions
help:
    @just --list

# Install all dependencies (including dev) via uv and pyproject.toml
install:
    # Remove existing virtual environment if present
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

    # Create fresh virtual environment
    python -m venv .venv

    # Activate virtual environment
    .venv\Scripts\Activate.ps1

    # Upgrade pip and install uv
    pip install --upgrade pip
    pip install uv

    # Sync dependencies from pyproject.toml
    uv sync --dev

# Run the main application using uv
run:
    uv run python src/mallet_sync/main.py

# Clean only output directory and Python caches
clean:
    # Remove output directory if exists
    if (Test-Path output) { Remove-Item -Recurse -Force output }

    # Remove Python cache directories
    Get-ChildItem -Recurse -Include __pycache__,*.pyc,*.pyo -Directory | Remove-Item -Recurse -Force

# Clean everything: venv, output, build artifacts, caches
clean-all:
    # Remove virtual environment
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

    # Remove output directory
    if (Test-Path output) { Remove-Item -Recurse -Force output }

    # Remove build artifacts
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    if (Test-Path build) { Remove-Item -Recurse -Force build }

    # Remove Python cache directories
    Get-ChildItem -Recurse -Include __pycache__,*.pyc,*.pyo -Directory | Remove-Item -Recurse -Force

# Freeze current environment to requirements.txt using uv
freeze:
    uv pip freeze > requirements.txt
