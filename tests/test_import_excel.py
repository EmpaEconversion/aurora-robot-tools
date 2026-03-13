"""Test functions from import_excel.py."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from aurora_robot_tools.import_excel import fix_mixed_batches, main


def test_fix_mixed_batches() -> None:
    """Check automatic batch assignment works."""
    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "a", "a", "b", "b", "b", "a", "a", "a"],
            "Batch Number": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch Number"].to_numpy() == np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))

    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "a", "a", "b", "b", "b", "a", "a", "a"],
            "Batch Number": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch Number"].to_numpy() == np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))

    df = pd.DataFrame(
        {
            "Bottom Electrode": ["a", "b", "c", "b", "b", "b", "a", "b", "c"],
            "Batch Number": [1, 1, 1, 1, 1, 1, 3, 3, 3],
        }
    )
    fix_mixed_batches(df)
    assert all(df["Batch Number"].to_numpy() == np.array([1, 2, 3, 2, 2, 2, 4, 5, 6]))

    df = pd.DataFrame(
        {
            "Bottom Electrode": [None, None, "a", "b", "b", "b", "a", "b", "c"],
            "Batch Number": [None, None, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    fix_mixed_batches(df)
    assert np.array_equal(
        df["Batch Number"].to_numpy(),
        np.array([np.nan, np.nan, 1, 2, 2, 3, 4, 3, 5]),
        equal_nan=True,
    )


def test_excel_import(test_dir: Path, tmp_path: Path) -> None:
    """Test full import."""
    fake_xlsx = test_dir / "date_name_tag.xlsx"
    with patch("aurora_robot_tools.import_excel.filedialog.askopenfilename", return_value=str(fake_xlsx)):
        main()
        db_path = tmp_path / "test.db"
        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            db_map = {}
            tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for (table,) in tables:
                cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
                col_names = [row[1] for row in cols]
                db_map[table] = col_names
        assert "Cell_Assembly_Table" in db_map
        assert "Electrolyte_Table" in db_map
        assert "Timestamp_Table" in db_map
        assert set(db_map["Timestamp_Table"]) == {"Cell Number", "Step Number", "Timestamp", "Complete"}

        assert "Settings_Table" in db_map
        assert "Calibration_Table" in db_map
        assert set(db_map["Calibration_Table"]) == {"Cell Number", "Step Number", "Rack Position", "dx_mm", "dy_mm"}
