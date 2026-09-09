"""Gamma1 V2：只修正V1连续缺失导致阈值永远为空的问题。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

import research.option_smirk_gamma1_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_option_smirk_gamma1_v2.yaml"


def load_config() -> dict:
    override = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base_path = ROOT / override["protocol"]["base_config_path"]
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config["protocol"] = {
        **config["protocol"],
        **override["protocol"],
    }
    config["signal"] = {
        **config["signal"],
        **override["signal_override"],
    }
    config["paths"] = override["paths"]
    config["governance"] = override["governance"]
    return config


def apply_valid_observation_threshold(daily: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = daily.copy()
    result["gamma1_lower_threshold"] = float("nan")
    result["buy_call_signal"] = False
    valid_mask = result["gamma1_30d"].notna()
    valid = result.loc[valid_mask, ["trade_date", "gamma1_30d"]].copy()
    window = int(config["signal"]["trailing_valid_observations"])
    lag = int(config["signal"]["threshold_lag_valid_observations"])
    quantile = float(config["signal"]["lower_quantile"])
    valid["gamma1_lower_threshold"] = (
        valid["gamma1_30d"].shift(lag).rolling(window, min_periods=window).quantile(quantile)
    )
    valid["buy_call_signal"] = valid["gamma1_30d"].le(
        valid["gamma1_lower_threshold"]
    ) & valid["gamma1_lower_threshold"].notna()
    threshold_map = valid.set_index("trade_date")["gamma1_lower_threshold"]
    signal_map = valid.set_index("trade_date")["buy_call_signal"]
    result.loc[valid_mask, "gamma1_lower_threshold"] = result.loc[
        valid_mask, "trade_date"
    ].map(threshold_map)
    result.loc[valid_mask, "buy_call_signal"] = (
        result.loc[valid_mask, "trade_date"].map(signal_map).fillna(False).astype(bool)
    )
    return result


def build_smirk_features(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    base_features = v1.build_smirk_features(panel, config)
    return apply_valid_observation_threshold(base_features, config)


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_smirk_gamma1_v2 import verify_protocol

    config, manifest = verify_protocol()
    v1.assert_input_hashes(config)
    panel, benchmark = v1.load_development_data(config)
    features = build_smirk_features(panel, config)
    result = v1.run_development(config, panel, benchmark, features)
    v1.write_outputs(result, features, config, manifest)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "ledger"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
