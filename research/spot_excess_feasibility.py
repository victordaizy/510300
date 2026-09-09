"""评估仅510300现货/现金在20日决策下达到年化超额20个百分点所需方向能力。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
TARGET_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
BENCHMARK_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_spot_excess_feasibility.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cagr(growth: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = (end - start).days / 365.2425
    return float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else np.nan


def build_non_overlapping_blocks(
    targets: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon: int = 20,
) -> dict[int, pd.DataFrame]:
    data = targets[
        ["date", f"exec_total_return_{horizon}d_net", f"label_end_date_{horizon}d"]
    ].copy()
    data["date"] = pd.to_datetime(data["date"])
    data[f"label_end_date_{horizon}d"] = pd.to_datetime(data[f"label_end_date_{horizon}d"])
    data = data.dropna().sort_values("date").reset_index(drop=True)
    total_return = benchmark[["date", "close"]].copy()
    total_return["date"] = pd.to_datetime(total_return["date"])
    close_map = total_return.set_index("date")["close"]
    blocks: dict[int, pd.DataFrame] = {}
    for offset in range(horizon):
        subset = data.iloc[offset::horizon].copy()
        subset["benchmark_start"] = subset["date"].map(close_map)
        subset["benchmark_end"] = subset[f"label_end_date_{horizon}d"].map(close_map)
        subset = subset.dropna(subset=["benchmark_start", "benchmark_end"])
        subset["benchmark_block_return"] = subset["benchmark_end"] / subset["benchmark_start"] - 1.0
        subset["positive_block"] = subset[f"exec_total_return_{horizon}d_net"].gt(0)
        if len(subset) >= 20:
            blocks[offset] = subset.reset_index(drop=True)
    return blocks


def simulate_required_accuracy(
    blocks: dict[int, pd.DataFrame],
    accuracies: list[float],
    repetitions: int = 2000,
    random_seed: int = 20260813,
    horizon: int = 20,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(random_seed)
    output: list[dict[str, float]] = []
    return_column = f"exec_total_return_{horizon}d_net"
    for accuracy in accuracies:
        excess_values: list[float] = []
        strategy_values: list[float] = []
        for subset in blocks.values():
            actual_positive = subset["positive_block"].to_numpy(dtype=bool)
            block_returns = subset[return_column].to_numpy(dtype=float)
            start = pd.Timestamp(subset["date"].iloc[0])
            end = pd.Timestamp(subset[f"label_end_date_{horizon}d"].iloc[-1])
            benchmark_growth = float(np.prod(1.0 + subset["benchmark_block_return"]))
            benchmark_cagr = _cagr(benchmark_growth, start, end)
            random_draws = rng.random((repetitions, len(subset)))
            correct = random_draws < accuracy
            predicted_positive = np.where(correct, actual_positive, ~actual_positive)
            strategy_growth = np.prod(
                np.where(predicted_positive, 1.0 + block_returns, 1.0), axis=1
            )
            years = (end - start).days / 365.2425
            strategy_cagr = np.power(strategy_growth, 1.0 / years) - 1.0
            strategy_values.extend(strategy_cagr.tolist())
            excess_values.extend((strategy_cagr - benchmark_cagr).tolist())
        output.append(
            {
                "symmetric_direction_accuracy": accuracy,
                "median_strategy_cagr": float(np.median(strategy_values)),
                "median_excess_cagr": float(np.median(excess_values)),
                "excess_cagr_10pct": float(np.quantile(excess_values, 0.10)),
                "excess_cagr_90pct": float(np.quantile(excess_values, 0.90)),
                "probability_excess_at_least_20pp": float(
                    np.mean(np.asarray(excess_values) >= 0.20)
                ),
            }
        )
    return output


def oracle_summary(
    blocks: dict[int, pd.DataFrame], horizon: int = 20
) -> dict[str, float]:
    return_column = f"exec_total_return_{horizon}d_net"
    oracle_excess: list[float] = []
    always_excess: list[float] = []
    for subset in blocks.values():
        start = pd.Timestamp(subset["date"].iloc[0])
        end = pd.Timestamp(subset[f"label_end_date_{horizon}d"].iloc[-1])
        benchmark_cagr = _cagr(
            float(np.prod(1.0 + subset["benchmark_block_return"])), start, end
        )
        oracle_growth = float(
            np.prod(
                np.where(subset["positive_block"], 1.0 + subset[return_column], 1.0)
            )
        )
        always_growth = float(np.prod(1.0 + subset[return_column]))
        oracle_excess.append(_cagr(oracle_growth, start, end) - benchmark_cagr)
        always_excess.append(_cagr(always_growth, start, end) - benchmark_cagr)
    return {
        "median_clairvoyant_oracle_excess_cagr": float(np.median(oracle_excess)),
        "minimum_clairvoyant_oracle_excess_cagr_across_offsets": float(np.min(oracle_excess)),
        "median_always_invested_roundtrip_excess_cagr": float(np.median(always_excess)),
    }


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    blocks = build_non_overlapping_blocks(
        pd.read_parquet(TARGET_FILE), pd.read_parquet(BENCHMARK_FILE)
    )
    if len(blocks) < 15:
        raise ValueError("可用非重叠20日错位组不足15组")
    accuracy_grid = [round(value, 2) for value in np.arange(0.50, 1.01, 0.05)]
    simulations = simulate_required_accuracy(blocks, accuracy_grid)
    required = next(
        (
            item["symmetric_direction_accuracy"]
            for item in simulations
            if item["median_excess_cagr"] >= settings["objective"]["annual_excess_target"]
        ),
        None,
    )
    report = {
        "status": "DIAGNOSTIC_ONLY_NOT_A_STRATEGY",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "scope": "510300现货或现金、20交易日非重叠决策、无杠杆无卖空",
        "method": "将真实20日净收益正负视为未知标签，以给定对称方向准确率随机翻转；每个准确率2000次、20个错位非重叠组。",
        "important_limitations": [
            "这是使用未来真实标签的能力需求模拟，不是可交易策略。",
            "假设正负样本识别准确率相同且错误独立，现实模型通常不满足。",
            "每20日强制平仓/重开，成本处理偏保守；现金收益设为0。",
            "全部历史已参与研究，结果只说明目标难度。",
        ],
        "non_overlapping_offset_count": len(blocks),
        "oracle": oracle_summary(blocks),
        "accuracy_simulation": simulations,
        "minimum_grid_accuracy_for_median_20pp_excess": required,
        "target_annual_excess": settings["objective"]["annual_excess_target"],
        "trading_use_authorized": False,
        "input_hashes": {
            TARGET_FILE.relative_to(ROOT).as_posix(): _sha256(TARGET_FILE),
            BENCHMARK_FILE.relative_to(ROOT).as_posix(): _sha256(BENCHMARK_FILE),
        },
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
