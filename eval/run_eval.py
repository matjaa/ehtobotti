"""Run the evaluation: `python eval/run_eval.py [options]`.

Same as `python -m ehtobotti eval`; kept here so the eval lives next to its question set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ehtobotti.cli import app  # noqa: E402

if __name__ == "__main__":
    app(["eval", *sys.argv[1:]])
