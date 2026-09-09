"""将已有慢速模型目标转换为2万元五档纸面库存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.small_account_execution import build_small_account_paper_target


CONFIG_FILE = PROJECT_ROOT / "config" / "small_account_20000.yaml"
OUTPUT_FILE = PROJECT_ROOT / "paper" / "510300_small_account_latest_signal.json"


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成510300两万元账户纸面目标")
    parser.add_argument("--current-shares", type=int, default=None, help="当前实际持有份额")
    parser.add_argument("--available-cash", type=float, default=None, help="当前可用现金")
    parser.add_argument("--sellable-shares", type=int, default=None, help="当前可卖旧仓份额")
    parser.add_argument("--today-bought-shares", type=int, default=None, help="当日买入且T+1不可卖份额")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    base_signal_file = _resolve(config["inputs"]["base_signal_file"])
    iopv_file = _resolve(config["inputs"]["iopv_snapshot_file"])
    readiness_file = _resolve(config["inputs"]["primary_market_readiness_report"])
    if not base_signal_file.exists() or not iopv_file.exists():
        print("缺少基础信号或IOPV前向快照，请先运行采集脚本", file=sys.stderr)
        return 1

    base_signal = json.loads(base_signal_file.read_text(encoding="utf-8"))
    readiness = (
        json.loads(readiness_file.read_text(encoding="utf-8"))
        if readiness_file.exists()
        else {"status": "NOT_EVALUATED", "eligible_for_position_mapping": False}
    )
    iopv = pd.read_parquet(iopv_file).sort_values("exchange_timestamp").iloc[-1]
    quote_trade_date = pd.Timestamp(iopv["trade_date"]).date().isoformat()
    if str(base_signal["signal_date"]) != quote_trade_date:
        print(
            f"基础信号日期{base_signal['signal_date']}与IOPV交易日{quote_trade_date}不一致，禁止计算交易差额",
            file=sys.stderr,
        )
        return 1
    current_shares = args.current_shares
    if current_shares is None:
        configured = config["account"].get("current_shares")
        current_shares = int(configured) if configured is not None else None
    available_cash = args.available_cash
    if available_cash is None:
        configured = config["account"].get("available_cash_cny")
        available_cash = float(configured) if configured is not None else None
    sellable_shares = args.sellable_shares
    if sellable_shares is None:
        configured = config["account"].get("sellable_shares")
        sellable_shares = int(configured) if configured is not None else None
    today_bought_shares = args.today_bought_shares
    if today_bought_shares is None:
        configured = config["account"].get("today_bought_shares")
        today_bought_shares = int(configured) if configured is not None else None
    target = build_small_account_paper_target(
        account_equity_cny=float(config["account"]["reference_equity_cny"]),
        raw_target_position=float(base_signal["target_position"]),
        price_cny=float(iopv["last_price"]),
        position_step=float(config["execution"]["position_grid_step"]),
        lot_size=int(config["execution"]["lot_size"]),
        minimum_normal_trade_shares=int(config["execution"]["minimum_normal_trade_shares"]),
        current_shares=current_shares,
        available_cash_cny=available_cash,
        sellable_shares=sellable_shares,
        today_bought_shares=today_bought_shares,
        allow_position_increase=bool(base_signal.get("allow_position_increase", False)),
        allow_position_decrease=bool(base_signal.get("allow_position_decrease", False)),
        commission_rate=float(config["execution"]["commission_rate"]),
        minimum_commission_cny=float(config["execution"]["minimum_commission_cny"]),
    )
    payload = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strategy": config["strategy"],
        "base_signal_date": base_signal["signal_date"],
        "quote_exchange_timestamp": str(pd.Timestamp(iopv["exchange_timestamp"])),
        "reference_price_cny": float(iopv["last_price"]),
        "reference_iopv": float(iopv["iopv"]),
        "premium_discount_bps": float(iopv["premium_discount_bps"]),
        "base_model": {
            "continuous_model_position": float(base_signal["continuous_model_position"]),
            "target_position": float(base_signal["target_position"]),
            "signal_reason": str(base_signal["signal_reason"]),
        },
        "execution_permission": {
            "trade_allowed": bool(base_signal.get("trade_allowed", False)),
            "is_review_day": bool(base_signal.get("is_review_day", False)),
            "risk_off_override": bool(base_signal.get("risk_off_override", False)),
            "allow_position_increase": bool(base_signal.get("allow_position_increase", False)),
            "allow_position_decrease": bool(base_signal.get("allow_position_decrease", False)),
        },
        "small_account_target": target,
        "primary_market_overlay": {
            **config["primary_market_overlay"],
            "readiness_status": readiness["status"],
            "full_coverage_days": int(readiness.get("full_coverage_days", 0)),
            "minimum_full_coverage_days": int(
                readiness.get("minimum_full_coverage_days", 20)
            ),
            "position_delta": 0.0,
            "reason": "PCF与IOPV只处于前向数据积累期，尚未通过仓位映射门槛",
        },
        "cost_assumptions": {
            "minimum_commission_cny": float(config["execution"]["minimum_commission_cny"]),
            "commission_rate": float(config["execution"]["commission_rate"]),
            "slippage_bps_per_leg": float(config["execution"]["slippage_bps_per_leg"]),
        },
        "governance": config["governance"],
        "input_hashes": {
            base_signal_file.name: _sha256(base_signal_file),
            iopv_file.name: _sha256(iopv_file),
            CONFIG_FILE.name: _sha256(CONFIG_FILE),
            **({readiness_file.name: _sha256(readiness_file)} if readiness_file.exists() else {}),
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
