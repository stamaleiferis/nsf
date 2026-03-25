#!/usr/bin/env python3
"""NSF Signal Separation — Desktop App

Run: python -m desktop.main
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.gui import run

if __name__ == "__main__":
    run()
