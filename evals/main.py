#!/usr/bin/env python3
"""
Multi-Domain PydanticAI Evaluation Suite
========================================

Main entry point for running evaluation scenarios.

Usage:
    python -m evals.main --list                    # List all scenarios
    python -m evals.main --scenario S01            # Run scenario S01
    python -m evals.main --scenario all            # Run all scenarios
    python -m evals.main --scenario S01 --model anthropic:claude-3.5-sonnet
"""

import sys
from .evaluator import main

if __name__ == "__main__":
    sys.exit(main())