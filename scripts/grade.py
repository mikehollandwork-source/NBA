"""Grade every locked lean whose game is final -> reports/live/record.csv + summary.
    python scripts/grade.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nba import record  # noqa: E402

led = record.grade_all()
kled = record.grade_kalshi()
wled = record.grade_winners()
txt = record.report(led, kled, wled)
(record.ROOT / "reports" / "live" / "record_summary.md").write_text(txt + "\n")
print(txt)
from nba import best_play  # noqa: E402
print("Top plays:", best_play.summary(best_play.grade()))
print("💰 Value (paper):", record.summarize_v(record.grade_value(wled)))
print("📏 ATS (paper):", record.summarize_v(record.grade_ats(wled)))
