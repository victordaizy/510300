"""LC + FVG + PV 策略原型的终端入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from logic import BacktestResult, run_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROTOTYPE_ROOT / "config.yaml"

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_state() -> tuple[dict, BacktestResult]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_file = PROJECT_ROOT / config["prototype"]["data_file"]
    minute = pd.read_parquet(data_file)
    return config, run_backtest(minute, config)


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value * 100:.2f}%"


def summary_text(result: BacktestResult) -> str:
    display = result.summaries.copy()
    for column in (
        "胜率",
        "平均毛收益",
        "平均净收益",
        "累计净收益",
        "年化净收益",
        "最大回撤",
    ):
        display[column] = display[column].map(format_percentage)
    display["盈利因子"] = display["盈利因子"].map(
        lambda value: "--" if pd.isna(value) else f"{value:.2f}"
    )
    return display.to_string(index=False)


def trade_text(trades: pd.DataFrame, count: int = 12) -> str:
    if trades.empty:
        return "没有生成交易。"
    columns = [
        "direction",
        "signal_time",
        "entry_time",
        "exit_time",
        "exit_reason",
        "signal_relative_volume",
        "gross_return",
        "net_return",
    ]
    display = trades.loc[:, columns].tail(count).copy()
    display["signal_relative_volume"] = display["signal_relative_volume"].map(
        lambda value: f"{value:.2f}"
    )
    display["gross_return"] = display["gross_return"].map(format_percentage)
    display["net_return"] = display["net_return"].map(format_percentage)
    return display.to_string(index=False)


def feature_state_text(result: BacktestResult) -> str:
    features = result.features
    return "\n".join(
        [
            f"分钟记录：{len(features):,}",
            f"交易日：{features['trade_date'].nunique():,}",
            f"多头扫流动性：{int(features['bullish_sweep'].sum()):,}",
            f"空头扫流动性：{int(features['bearish_sweep'].sum()):,}",
            f"合格多头 FVG：{int(features['bullish_fvg_candidate'].sum()):,}",
            f"合格空头 FVG：{int(features['bearish_fvg_candidate'].sum()):,}",
            f"最终交易：{len(result.trades):,}",
        ]
    )


def render(config: dict, result: BacktestResult, view: str) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{BOLD}{config['prototype']['name']}{RESET}")
    print(f"{DIM}原型问题：{config['prototype']['question']}{RESET}\n")
    print(f"{BOLD}当前状态{RESET}")
    print(feature_state_text(result))
    print()
    if view == "summary":
        print(f"{BOLD}分段结果{RESET}")
        print(summary_text(result))
    elif view == "trades":
        print(f"{BOLD}最近交易{RESET}")
        print(trade_text(result.trades))
    elif view in {"long", "short"}:
        direction = "LONG" if view == "long" else "SHORT"
        print(f"{BOLD}{'多头' if view == 'long' else '空头'}交易{RESET}")
        print(trade_text(result.trades[result.trades["direction"] == direction]))
    print(
        f"\n{BOLD}[s]{RESET} {DIM}汇总{RESET}  "
        f"{BOLD}[t]{RESET} {DIM}最近交易{RESET}  "
        f"{BOLD}[l]{RESET} {DIM}多头{RESET}  "
        f"{BOLD}[k]{RESET} {DIM}空头{RESET}  "
        f"{BOLD}[q]{RESET} {DIM}退出{RESET}"
    )


def print_report(config: dict, result: BacktestResult) -> None:
    print(config["prototype"]["name"])
    print(f"原型问题：{config['prototype']['question']}")
    print(feature_state_text(result))
    print("\n分段结果")
    print(summary_text(result))
    print("\n最近交易")
    print(trade_text(result.trades))


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 LC + FVG + PV 策略原型")
    parser.add_argument(
        "--report",
        action="store_true",
        help="输出一次完整报告后退出，适用于自动化验证",
    )
    args = parser.parse_args()
    config, result = load_state()
    if args.report:
        print_report(config, result)
        return

    view = "summary"
    while True:
        render(config, result, view)
        command = input("\n请输入操作：").strip().lower()
        if command == "q":
            break
        view = {
            "s": "summary",
            "t": "trades",
            "l": "long",
            "k": "short",
        }.get(command, view)


if __name__ == "__main__":
    main()

