# mallet-sync

A Python-based audio synchronization tool.
This project uses [just](https://github.com/casey/just) for task automation and [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

---

## Prerequisites

Before using this project, please ensure you have the following tools installed:

- **just** (for running project tasks):
  [Install instructions](https://github.com/casey/just)

- **uv** (for dependency management):
  [Install instructions](https://github.com/astral-sh/uv#installation)

- **Python 3.13+** (required for the codebase)

---

## Setup & Usage

1. **Clone this repository**:

   ```shell
   git clone https://github.com/klugman-yanai/mallet-sync
   cd mallet-sync
   ```

2. **Install dependencies** (recommended: using uv and pyproject.toml):

   ```shell
   just install
   ```

   If you do not have uv, you may instead use pip and requirements.txt:

   ```shell
   just install-pip
   ```

3. **Activate the virtual environment** (PowerShell):

   ```shell
   just activate
   ```

4. **Run the main application:**

   ```shell
   just run
   ```

---

## Additional Commands

- **Clean output and cache files:**

  ```shell
  just clean
  ```

- **Remove all build artifacts, output, and the virtual environment:**

  ```shell
  just clean-all
  ```

- **Freeze the current environment to requirements.txt:**

  ```shell
  just freeze
  ```

- **See all available commands:**

  ```shell
  just --list
  ```
