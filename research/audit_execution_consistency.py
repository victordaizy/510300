"""审计注册因子Target与全部实际回测台账的执行一致性。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
DATASET_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
MARKET_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
TRADE_ROOTS = (
    ROOT / "data" / "processed" / "registered_factor_backtests",
    ROOT / "data" / "processed" / "round2_mechanism_backtests",
    ROOT / "data" / "processed" / "round3_breadth_backtests",
    ROOT / "data" / "processed" / "round4_cap_weighted_backtests",
)
JSON_FILE = ROOT / "reports" / "research" / "execution_consistency_audit.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "execution_consistency_audit.md"


def _commission(notional: float, settings: dict) -> float:
    if notional <= 0:
        return 0.0
    return max(float(settings["minimum_commission_cny"]), notional * float(settings["commission_rate"]))


def main() -> int:
    for path in (SETTINGS_FILE, DATASET_FILE, MARKET_FILE, DIVIDEND_FILE):
        if not path.exists():
            raise FileNotFoundError(f"执行一致性审计缺少输入：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    costs = settings["backtest"]
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["date"] = pd.to_datetime(dataset["date"])
    market = pd.read_parquet(MARKET_FILE).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"])
    market_index = {date: index for index, date in enumerate(market["date"])}
    open_by_date = market.set_index("date")["open"].astype(float).to_dict()
    failures: list[str] = []
    if not dataset["signal_asof_date"].pipe(pd.to_datetime).equals(dataset["date"]):
        failures.append("因子信息截止日与信号日不一致")
    label_rows_checked = 0
    for horizon in (5, 20):
        end_column = f"label_end_date_{horizon}d"
        for row in dataset.loc[dataset[end_column].notna(), ["date", end_column]].itertuples(index=False):
            signal_date = pd.Timestamp(row[0])
            end_date = pd.Timestamp(row[1])
            if signal_date not in market_index or end_date not in market_index:
                failures.append(f"{horizon}日标签日期不在ETF交易日历：{signal_date.date()}")
                continue
            if market_index[end_date] - market_index[signal_date] != horizon:
                failures.append(f"{horizon}日标签终点偏移错误：{signal_date.date()}->{end_date.date()}")
            label_rows_checked += 1
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    non_market_payment_dates = [
        str(date.date()) for date in dividends["payment_date"] if pd.Timestamp(date) not in market_index
    ]
    if non_market_payment_dates:
        failures.append(f"分红发放日不在交易日历：{non_market_payment_dates}")
    trade_files = sorted(path for root in TRADE_ROOTS if root.exists() for path in root.rglob("*_trades.parquet"))
    if not trade_files:
        failures.append("没有找到注册因子回测交易台账")
    trade_rows_checked = 0
    ledger_files_checked = 0
    slippage = float(costs["slippage_bps_base"]) / 10_000.0
    lot_size = int(costs["lot_size"])
    for path in trade_files:
        trades = pd.read_parquet(path)
        if trades.empty:
            continue
        ledger_files_checked += 1
        trades["date"] = pd.to_datetime(trades["date"])
        trades["signal_date"] = pd.to_datetime(trades["signal_date"])
        for row in trades.itertuples(index=False):
            date = pd.Timestamp(row.date)
            signal_date = pd.Timestamp(row.signal_date)
            if date not in market_index or signal_date not in market_index:
                failures.append(f"交易或信号日期不在交易日历：{path.name}")
                continue
            if market_index[date] - market_index[signal_date] != 1:
                failures.append(f"交易未使用下一交易日开盘：{path.name} {signal_date.date()}->{date.date()}")
            if int(row.quantity) % lot_size != 0:
                failures.append(f"交易数量不是整手：{path.name} {date.date()}")
            if row.side == "卖出" and int(row.quantity) > int(row.t_plus_one_sellable_before_trade):
                failures.append(f"卖出超过T+1可用份额：{path.name} {date.date()}")
            expected_price = float(open_by_date[date]) * (1.0 + slippage if row.side == "买入" else 1.0 - slippage)
            if not np.isclose(float(row.execution_price), expected_price, rtol=0.0, atol=1e-10):
                failures.append(f"滑点成交价错误：{path.name} {date.date()}")
            expected_commission = _commission(int(row.quantity) * float(row.execution_price), costs)
            if not np.isclose(float(row.commission), expected_commission, rtol=0.0, atol=1e-8):
                failures.append(f"佣金错误：{path.name} {date.date()}")
            trade_rows_checked += 1
    ledger_paths = sorted(path for root in TRADE_ROOTS if root.exists() for path in root.rglob("*_ledger.parquet"))
    negative_receivable_rows = 0
    for path in ledger_paths:
        ledger = pd.read_parquet(path)
        if not ledger.empty and "dividend_receivable" in ledger:
            negative_receivable_rows += int((ledger["dividend_receivable"] < -1e-8).sum())
    if negative_receivable_rows:
        failures.append(f"分红应收出现负值：{negative_receivable_rows}行")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "label_rows_checked": label_rows_checked,
        "trade_files_found": len(trade_files),
        "nonempty_trade_files_checked": ledger_files_checked,
        "trade_rows_checked": trade_rows_checked,
        "ledger_files_checked": len(ledger_paths),
        "dividend_events_checked": int(len(dividends)),
        "checks": {
            "signal_asof_not_future": True,
            "label_end_offset_matches_horizon": True,
            "trade_uses_next_market_open": True,
            "sell_not_above_t1_available": True,
            "lot_size": lot_size,
            "execution_slippage_bps": float(costs["slippage_bps_base"]),
            "commission_rate": float(costs["commission_rate"]),
            "minimum_commission_cny": float(costs["minimum_commission_cny"]),
            "dividend_payment_dates_on_market_calendar": True,
            "dividend_receivable_never_negative": True,
        },
        "failures": failures,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Target与回测执行一致性审计", "",
        f"- 状态：`{report['status']}`。",
        f"- 检查可执行标签：{label_rows_checked}行。",
        f"- 检查实际交易：{trade_rows_checked}笔，来自{ledger_files_checked}个非空台账。",
        f"- 检查净值台账：{len(ledger_paths)}个；分红事件：{len(dividends)}个。",
        "- 逐项验证下一交易日开盘、T+1可用份额、整手、滑点成交价、佣金、分红发放日和应收股利非负。", "",
    ]
    if failures:
        lines += ["## 失败项", "", *[f"- {item}" for item in failures], ""]
    MARKDOWN_FILE.write_text("\n".join(lines), encoding="utf-8")
    if failures:
        raise RuntimeError(f"执行一致性审计失败：{failures[:5]}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
