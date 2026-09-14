# =============================================================================
# MODULE: run_scaffold_cv.py
# PURPOSE: Preserve the original scaffold-CV command while using the generic runner.
# LIBRARIES: The generic run_openbind_cv module contains all orchestration logic.
# OUTPUT: Identical to `python run_openbind_cv.py --cv_method scaffold`.
# CALCULATIONS: No model calculations occur in this compatibility entry point.
# =============================================================================
"""Backward-compatible scaffold-only entry point for the generic OpenBind CV runner."""

from __future__ import annotations  # Postpone type-annotation evaluation consistently with the project.

import sys  # Insert the scaffold method into command-line arguments when omitted.

from run_openbind_cv import main  # Reuse the single tested random/scaffold orchestration implementation.


if __name__ == "__main__":  # Execute only when the compatibility script is invoked directly.
    if "--cv_method" not in sys.argv:  # Preserve historical scaffold behaviour unless the user explicitly chooses a method.
        sys.argv.extend(["--cv_method", "scaffold"])  # Supply the generic runner's scaffold selection option.
    main()  # Generate scaffold folds, run selected models and aggregate results.
