"""RetinaSight Root Test Runner Forwarder.

Redirects execution to the organized test suite under `tests/test_preprocessing.py`.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_preprocessing import run_tests

if __name__ == "__main__":
	run_tests()
