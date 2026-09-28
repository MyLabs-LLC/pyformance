"""Launch PyFormance.

    python benchmark.py
    python benchmark.py --cli
    python benchmark.py --cli --quick
"""

import sys
from pathlib import Path

_package_dir = Path(__file__).resolve().parent
if _package_dir.name == "pyformance":
    sys.path.insert(0, str(_package_dir.parent))

from pyformance.__main__ import main

if __name__ == "__main__":
    main()
