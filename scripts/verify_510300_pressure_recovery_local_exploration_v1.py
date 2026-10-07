"""复算保存的本地探索结果；不重选事件、不扩展参数或联网取数。"""

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_index(path, files, relative_to):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows({"path": item.relative_to(relative_to).as_posix(), "bytes": item.stat().st_size, "sha256": sha(item)} for item in sorted(files, key=str))


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    registration = json.loads((OUT / "registration_receipt.json").read_text(encoding="utf-8"))
    grid = pd.read_parquet(OUT / "01_盘口与消息测量表.parquet")
    events = pd.read_csv(OUT / "02_固定观察事件表.csv")
    quotes = pd.read_csv(OUT / "03_可见盘口数量与费用表.csv")
    scenarios = pd.read_csv(OUT / "04_跨日价格费用假设表.csv")
    checks = {}
    checks["登记输入及原合同保持原样"] = all(sha(ROOT / row["path"]) == row["sha256"] for row in registration["files"])
    checks["完整分钟网格与实际合格窗口"] = len(grid) == 711 and grid.log_mid_return_5m_bps.notna().sum() == 679
    checks["事件不按反弹结果删选"] = len(events) == 17 and (events.observation_price_change_bps.gt(0).sum(), events.observation_price_change_bps.le(0).sum()) == (7, 10)
    start = pd.to_datetime(events.shock_at)
    end = pd.to_datetime(events.observation_end_at)
    entry = pd.to_datetime(events.hypothetical_entry_quote_at)
    checks["固定观察终点与严格之后的报价"] = ((end - start) == pd.Timedelta(minutes=5)).all() and ((entry - end) > pd.Timedelta(0)).all() and ((entry - end) <= pd.Timedelta(seconds=5)).all()
    observed = grid.source_quote_at.notna()
    checks["数据没有未来报价"] = grid.quote_age_seconds.dropna().between(0, 5).all() and (pd.to_datetime(grid.loc[observed, "source_quote_at"]).to_numpy() <= grid.loc[observed, "decision_at"].to_numpy()).all()
    checks["报价数量和现金预算"] = len(quotes) == 1418 and quotes.quantity.mod(100).eq(0).all() and (quotes.cash_required_cny <= quotes.nominal_budget_cny + 1e-8).all()
    values = scenarios.loc[scenarios.exit_minute_proxy_cny.notna()]
    buy = values.quantity * values.hypothetical_entry_price_cny
    sell = values.quantity * values.exit_minute_proxy_cny * (1 - 0.0005)
    fees = np.maximum(buy * 0.0002, 5) + np.maximum(sell * 0.0002, 5)
    recomputed = sell - buy - fees
    checks["条件价格费用算术复算"] = np.allclose(recomputed, values.hypothetical_cash_difference_cny, atol=2e-6, rtol=1e-10) and np.allclose(recomputed / buy * 10000, values.hypothetical_cash_difference_bps, atol=2e-7, rtol=1e-10)
    checks["跨日缺失没有补成成交"] = scenarios.loc[scenarios.trade_date.eq("2026-09-04"), "hypothetical_cash_difference_bps"].isna().all()
    checks["八月盘后无记录保留为未知"] = next(row for row in summary["close_paths"] if row["trade_date"] == "2026-08-12")["after_1506_source_volume"] is None
    checks["短时路径汇总复算"] = all(math.isclose(float(events[f"post_entry_mid_path_{horizon}m_bps"].mean()), summary[f"post_entry_{horizon}m_mid_price_path_bps"]["mean"], rel_tol=1e-10, abs_tol=1e-9) for horizon in (5, 30))
    checks["假设不变成原策略收益"] = scenarios.verified_strategy_return.eq(False).all() and summary["actual_orders"] == 0 and summary["full_account_sharpe"] == "NOT_COMPUTED" and summary["goal_achieved"] is False
    checks["图已生成且结果目录低于20MiB"] = (OUT / "三日盘口与固定观察窗口.png").exists() and sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) < 20 * 1024 ** 2
    checks = {name: bool(value) for name, value in checks.items()}
    result = {"verified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "tests": {"new_exploration_tests": 6, "existing_measurement_tests": 21, "total_passed": 27, "failed": 0,
                        "evidence": "本任务实际测试输出；本脚本不重复运行测试。"},
              "scope": "计算、输入身份和情景标记的本地复核；不代表可成交、统计显著或策略成立。"}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    code = [ROOT / name for name in (
        "config/510300_pressure_recovery_local_exploration_v1.json",
        "research/pressure_recovery_local_exploration_v1.py",
        "scripts/run_510300_pressure_recovery_local_exploration_v1.py",
        "scripts/verify_510300_pressure_recovery_local_exploration_v1.py",
        "tests/test_510300_pressure_recovery_local_exploration_v1.py")]
    make_index(OUT / "FILE_INDEX.csv", [path for path in OUT.rglob("*") if path.is_file() and path.name != "FILE_INDEX.csv"] + code, ROOT)
    parent = OUT.parent
    make_index(parent / "FILE_INDEX.csv", [path for path in parent.rglob("*") if path.is_file() and path != parent / "FILE_INDEX.csv"], parent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("最终结果目录字节数：", sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))


if __name__ == "__main__":
    main()
