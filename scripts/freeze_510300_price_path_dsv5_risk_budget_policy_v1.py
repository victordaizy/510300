"""仅字节身份和模式检查，不读取预测数值或市场价格。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
import yaml
from research.price_path_dsv5_risk_budget_policy_v1 import (
    CONFIG, MANIFEST, MODEL_ID, REPORT, SLUG, ContractError,
    git, identity, load_config, now, read_json, write_new,
)


def freeze(root: Path) -> dict:
    cfg = load_config(root)
    if git(root, "branch", "--show-current") != cfg["required_branch"]:
        raise ContractError("冻结分支与合同不一致")
    receipt = read_json(root / cfg["inputs"]["parent_receipt"])
    expected = {x["path"]: x for x in receipt["outputs"]}
    inputs = {name: identity(root, path) for name, path in cfg["inputs"].items()}
    for name in ["predictions", "coefficients", "parent_result", "parent_status"]:
        if inputs[name] != expected[cfg["inputs"][name]]:
            raise ContractError(f"父实验回执哈希不符：{name}")
    parent_cfg = yaml.safe_load((root / cfg["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    for key, parent_key in [("prices", "etf_unadjusted_daily"), ("dividends", "etf_cash_dividends")]:
        expected_identity = {k: parent_cfg["inputs"][parent_key][k] for k in ["path", "bytes", "sha256"]}
        if inputs[key] != expected_identity:
            raise ContractError(f"父市场输入字节已改变：{key}")
    source_schema = pq.ParquetFile(root / cfg["inputs"]["predictions"]).schema_arrow.names
    if not set(cfg["prediction_projection"]).issubset(source_schema):
        raise ContractError("父预测模式缺少所需字段")
    implementation = [CONFIG,
        "docs/510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1_PROTOCOL_20260905.md",
        "docs/510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1_USER_REQUEST_20260905.md",
        "docs/510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1_APPEND_ONLY_CLOSURE_20260905.md",
        "research/price_path_dsv5_risk_budget_policy_v1.py",
        f"scripts/freeze_{SLUG}.py", f"scripts/run_{SLUG}.py",
        f"tests/test_{SLUG}.py"]
    prior = sorted(p.relative_to(root).as_posix() for p in (root / "config").glob("*manifest*.json")
                   if p.relative_to(root).as_posix() != MANIFEST)
    inventory = {"manifest_count_before_current": len(prior),
        "manifest_count_including_current": len(prior) + 1,
        "scope": "ALL_PROJECT_CONFIG_MANIFESTS_INCLUDING_NON_510300",
        "complete_trial_budget_proven": False, "cross_trial_sr_variance_available": False,
        "prior_manifests": [identity(root, p) for p in prior]}
    result = {"MODEL_ID": MODEL_ID, "version": cfg["version"], "frozen_at": now(),
        "implementation": [identity(root, p) for p in implementation], "inputs": inputs,
        "trial_inventory": inventory, "prediction_rows_metadata_only": pq.ParquetFile(root / cfg["inputs"]["predictions"]).metadata.num_rows,
        "prediction_values_read": False, "market_price_values_read": False,
        "portfolio_return_values_read": False, "parent_trainer_called": False,
        "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False}
    write_new(root / MANIFEST, result)
    write_new(root / f"{REPORT}/freeze_receipt.json", {"MODEL_ID": MODEL_ID, "frozen_at": result["frozen_at"],
        "manifest_identity": identity(root, MANIFEST), "STATE": "FROZEN_AWAITING_GIT_COMMIT",
        "prediction_values_read": False, "portfolio_return_values_read": False})
    return result


if __name__ == "__main__":
    result = freeze(ROOT)
    print(f"冻结完成：{result['MODEL_ID']}；未读取预测数值或组合收益。")
