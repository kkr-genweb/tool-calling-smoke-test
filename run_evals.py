#!/usr/bin/env python3
"""
Multi-Domain PydanticAI Evaluation Suite
========================================

Standalone script for running evaluation scenarios.

Usage:
    python run_evals.py --list                    # List all scenarios
    python run_evals.py --scenario S01            # Run scenario S01
    python run_evals.py --scenario all            # Run all scenarios
    python run_evals.py --scenario S01 --model anthropic:claude-3.5-sonnet
"""

import sys
from evals.evaluator import main

if __name__ == "__main__":
    sys.exit(main())