from __future__ import annotations

from evaluate import main


"""Compatibility entry point.

Use evaluate.py for patch-level metrics, predict_single.py for one selected
patch, and predict_full_image.py for complete microscope-image prediction.
"""


if __name__ == "__main__":
    main()
