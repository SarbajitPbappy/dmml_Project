from __future__ import annotations

from typing import Literal

import numpy as np

from retrox.config import settings

ERSLevel = Literal["normal", "elevated", "high"]


def classify_ers(pred_cases: float | int) -> ERSLevel:
    """
    3-level alerting based on predicted weekly cases (PPT thresholds).
    - normal: < 39
    - elevated: 39..71
    - high: > 71
    """
    x = float(pred_cases)
    if not np.isfinite(x):
        return "normal"
    if x <= settings.ers_normal_max:
        return "normal"
    if x <= settings.ers_elevated_max:
        return "elevated"
    return "high"

