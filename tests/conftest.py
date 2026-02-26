import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
