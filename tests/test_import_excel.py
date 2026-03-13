"""Test functions from import_excel.py."""

import numpy as np
import pandas as pd

from aurora_robot_tools.import_excel import fix_mixed_batches


def test_fix_mixed_batches() -> None:
    """Check automatic batch assignment works."""
    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "a", "a", "b", "b", "b", "a", "a", "a"],
            "Batch": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch"].to_numpy() == np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))

    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "a", "a", "b", "b", "b", "a", "a", "a"],
            "Batch": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch"].to_numpy() == np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))

    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "b", "c", "b", "b", "b", "a", "b", "c"],
            "Batch": [1, 1, 1, 1, 1, 1, 3, 3, 3],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch"].to_numpy() == np.array([1, 2, 3, 2, 2, 2, 4, 5, 6]))
