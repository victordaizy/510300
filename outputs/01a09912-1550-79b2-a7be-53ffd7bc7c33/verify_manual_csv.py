"""仅用 Python 标准库复算同包 D60 数据，不访问网络或账户程序。"""
import csv
import json
import math
from pathlib import Path

base = Path(__file__).resolve().parent
csv_path = base / "数据" / "D60每日数据.csv"
if not csv_path.exists():
    csv_path = base.parents[1] / "reports/research/510300_daily_manual_signal_v1/2026-09-11/D60每日数据.csv"
rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
differences = []
previous_score = None
score_count = 0
max_error = 0.0
for row in rows:
    prior_text = row["previous_close"]
    if not prior_text:
        differences.append(None)
        continue
    previous_close, opening, close, cash = (float(row[k]) for k in ["previous_close", "open", "close", "cash_dividend_per_share"])
    overnight = math.log((opening + cash) / previous_close)
    intraday = math.log((close + cash) / (opening + cash))
    delta = intraday - overnight
    for key, value in [("overnight_log", overnight), ("intraday_log", intraday), ("difference", delta)]:
        assert abs(float(row[key]) - value) < 1e-12, (row["date"], key)
    differences.append(delta)
    window = differences[-60:]
    if len(window) < 60 or any(v is None for v in window):
        assert row["score60"] == ""
        continue
    total = math.fsum(window)
    mean = total / 60
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in window) / 59)
    if std == 0:
        assert row["score60"] == ""
        previous_score = None
        continue
    score = total / (std * math.sqrt(60))
    error = abs(score - float(row["score60"]))
    max_error = max(max_error, error)
    assert error < 1e-10, (row["date"], score, row["score60"])
    entry = int(previous_score is not None and score > 1 and previous_score > 1)
    exit_flag = int(previous_score is not None and score < 0 and previous_score < 0)
    assert int(row["entry_condition_2days"]) == entry, row["date"]
    assert int(row["difference_exit_condition_2days"]) == exit_flag, row["date"]
    previous_score = score
    score_count += 1
result = {"status": "PASS_STDLIB_OFFLINE_D60_RECOMPUTATION", "rows": len(rows), "score_rows": score_count,
          "max_absolute_error": max_error, "network_requests": 0, "new_accounts": 0, "model_fits": 0}
(base / "CSV离线复算回执.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
