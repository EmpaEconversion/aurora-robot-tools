"""High level integration tests."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from aurora_robot_tools.cli import app

runner = CliRunner()


def test_excel_import(test_dir: Path, tmp_path: Path) -> None:
    """Test full import."""
    fake_xlsx = test_dir / "date_name_tag.xlsx"

    with (
        patch("aurora_robot_tools.import_excel.Tk"),
        patch("aurora_robot_tools.import_excel.filedialog.askopenfilename", return_value=str(fake_xlsx)),
    ):
        result = runner.invoke(app, ["import-excel"])

    assert result.exit_code == 0

    # Check db now exists and has the correct structure
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

    # Assign cell numbers without weighing = no cells
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    with sqlite3.connect(db_path) as conn:
        res = conn.cursor().execute("SELECT `Cell Number` FROM Cell_Assembly_Table").fetchall()
        assert all(c == 0 for (c,) in res)

    # Add weights
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        anode_weights = np.linspace(18.7, 19.1, 24)
        cathode_weights = np.linspace(13.1, 12.8, 24)
        for i in range(0, 24):
            cur.execute(
                "UPDATE Cell_Assembly_Table SET `Anode Mass (mg)`= ? WHERE `Rack Position` = ?",
                (anode_weights[i], i + 1),
            )
            cur.execute(
                "UPDATE Cell_Assembly_Table SET `Cathode Mass (mg)`= ? WHERE `Rack Position` = ?",
                (cathode_weights[i], i + 1),
            )

    # Assign cell numbers with weights
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    with sqlite3.connect(db_path) as conn:
        res = conn.cursor().execute("SELECT `Cell Number`, `N:P Ratio` FROM Cell_Assembly_Table").fetchall()
        cell_ns = [x[0] for x in res]
        np_ratios = [x[1] for x in res]
    assert cell_ns == list(range(1, 25)) + [0] * 12
    assert all(np > 1.0 for np in np_ratios[:24])
    assert all(np < 1.2 for np in np_ratios[:24])

    # Calculate electrolyte mixing steps
    result = runner.invoke(app, ["electrolyte"])
    assert result.exit_code == 0
    with sqlite3.connect(db_path) as conn:
        res = (
            conn.cursor()
            .execute("SELECT `Source Position`, `Target Position`, `Volume (uL)` FROM Mixing_Table")
            .fetchall()
        )
        source = [x[0] for x in res]
        target = [x[1] for x in res]
        vol = [x[2] for x in res]
    assert source == sorted(source)
    assert all(t > s for t, s in zip(target, source, strict=True))
    assert all(v > 0 for v in vol)

    # Assign cells to press
    result = runner.invoke(app, ["assign", "False"])
    with sqlite3.connect(db_path) as conn:
        res = conn.cursor().execute("SELECT `Current Press Number` FROM Cell_Assembly_Table").fetchall()
        press_ns = [x[0] for x in res]
    assert press_ns == list(range(1, 7)) + [0] * 30
