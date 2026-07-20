"""``python -m retrieval`` entry point — delegates to the ``retrieval`` CLI."""

import sys

from retrieval.cli import main

if __name__ == "__main__":
    sys.exit(main())
