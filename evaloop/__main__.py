#!/usr/bin/env python3
"""
Main entry point for EvaLoop when called as a module.

This allows the package to be run with:
    python -m evaloop [command] [args]
"""

from .cli import main

if __name__ == "__main__":
    main()
