#!/bin/bash
# run_tests.sh - A foolproof runner for mlc-llm tests within the local .venv

set -e

PROJECT_ROOT="/workspace/projects/mlc-llm"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "--- Starting mlc-llm Test Runner ---"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH. Please create it first."
    exit 1
fi

# Use the absolute path to the python executable inside the venv
# This bypasss the need for 'source activate' which can be flaky in non-interactive shells
PYTHON_EXE="$VENV_PATH/bin/python"

if [ ! -x "$PYTHON_EXE" ]; then
    echo "ERROR: Python executable not found at $PYTHON_EXE. Is the venv corrupted?"
    exit 1
fi

echo "Using Python: $($PYTHON_EXE --version) ($PYTHON_EXE)"

# Add project root to PYTHONPATH so modules can be found regardless of CWD
export PYTHONPATH="$PROJECT_ROOT/python:$PYTHONPATH"

echo "Running pytest via $PYTHON_EXE -m pytest..."
exec "$PYTHON_EXE" -m pytest "$@"
