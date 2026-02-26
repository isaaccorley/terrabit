"""Type aliases."""

from __future__ import annotations

import numpy as np

type NDArrayF32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
