"""Convenience wrapper for src/run_baseline.py"""
import sys
from pathlib import Path

# Ensure root and src are on path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.run_baseline import main

if __name__ == "__main__":
    main()
