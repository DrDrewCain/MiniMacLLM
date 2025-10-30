"""
Pytest configuration file.

This file is automatically loaded by pytest and configures the test environment.
"""

import sys
from pathlib import Path

# Add the project root to Python path so we can import src modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
