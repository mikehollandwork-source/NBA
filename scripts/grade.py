"""Grade every locked lean whose game is final -> reports/live/record.csv + summary.
    python scripts/grade.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nba import record  # noqa: E402

led = record.grade_all()
kled = record.grade_kalshi()
txt = record.report(led, kled)
(record.ROOT / "reports" / "live" / "record_summary.md").write_text(txt + "\n")
print(txt)
