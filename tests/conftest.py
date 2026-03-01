"""
conftest.py

Adds the project root to sys.path so that test modules can import
team_analyser, logo_utils, gui, and main without installation.
"""

import sys
from pathlib import Path

# Insert the directory containing team_analyser.py, gui.py etc.
sys.path.insert(0, str(Path(__file__).parent.parent))
