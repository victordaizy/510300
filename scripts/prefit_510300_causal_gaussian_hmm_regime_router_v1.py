"""仅用2015年前数据拟合并登记510300因果HMM模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from causal_gaussian_hmm_regime_router_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
    ContractError,
    _project_path,
    build_prefreeze_audit,
    fit_prefreeze_model,
    load_config,
    sha256_file,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅用2012-06-26至2014-12-31拟合510300因果HMM"
    )
    parser.add_argument(
        "--overwrite-prefreeze",
        action="store_true",
        help="仅在冻结清单不存在时覆盖尚未冻结的模型与输入审计",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_config(CONFIG_PATH)
        audit_path = _project_path(config["paths"]["input_audit"])
        if MANIFEST_PATH.exists():
            raise ContractError("冻结清单已存在，拒绝重新拟合模型")
        existing = [path for path in (MODEL_PATH, audit_path) if path.exists()]
        if existing and not arguments.overwrite_prefreeze:
            raise FileExistsError(f"预冻结产物已存在，拒绝覆盖：{existing}")
        index_path = _project_path(config["inputs"]["index_daily"]["path"])
        if sha256_file(index_path) != config["inputs"]["index_daily"]["sha256"]:
            raise ContractError("000300训练输入哈希不符合合同")
        index = pd.read_parquet(index_path)
        model = fit_prefreeze_model(index, config)
        if model["status"] != "PASS_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY":
            payload = {
                "status": model["status"],
                "training_diagnostics": model["training_diagnostics"],
                "candidate_2015_plus_return_evaluation": "NOT_ALLOWED",
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit = build_prefreeze_audit(config, model, MODEL_PATH)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (ContractError, FileNotFoundError, FileExistsError, ValueError) as exc:
        payload = {
            "status": "PREFIT_FAILED_NO_2015_PLUS_VIEW",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_2015_plus_return_evaluation": "NOT_ALLOWED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "status": "PASS_PREFIT_READY_TO_FREEZE",
        "project_id": model["project_id"],
        "training_start": model["training"]["start"],
        "training_end": model["training"]["end"],
        "training_rows": model["training"]["rows"],
        "converged": model["training_diagnostics"]["converged"],
        "iterations": model["training_diagnostics"]["iterations"],
        "minimum_effective_state_mass": model["training_diagnostics"][
            "minimum_effective_state_mass"
        ],
        "state_mapping": model["state_mapping"],
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_sha256": sha256_file(MODEL_PATH),
        "input_audit_path": audit_path.relative_to(ROOT).as_posix(),
        "input_audit_sha256": sha256_file(audit_path),
        "candidate_2015_plus_states_computed": False,
        "candidate_2015_plus_outcomes_computed": False,
        "candidate_portfolio_returns_computed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
