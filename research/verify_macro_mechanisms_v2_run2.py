"""从已保存输出复核必要恒等式、标签复用和资料时钟，写入完成记录。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    frame = pd.read_csv(STUDY / "results/104个月_机制分解与原冻结标签.csv")
    parent = pd.read_csv(STUDY / "inputs/104个月_完整观察.csv").set_index("stat_month")
    assert len(frame) == 104 and frame.stat_month.is_unique
    by_month = frame.set_index("stat_month")
    identities = 0
    maximum_residual = 0.0
    for row in frame.to_dict("records"):
        for lag in [1, 3]:
            earlier = str(pd.Period(row["stat_month"], freq="M") - lag)
            expected = earlier in by_month.index and by_month.loc[earlier, "definition_version"] == row["definition_version"]
            assert bool(row[f"d{lag}_eligible"]) == expected
            if not expected:
                assert pd.isna(row[f"d{lag}_relative_growth_log_pp"])
                continue
            old = by_month.loc[earlier]
            current_terms, base_terms, growth_terms = [], [], []
            for metric in ["m1", "m2"]:
                now_level, old_level = row[f"{metric}_balance_100m"], old[f"{metric}_balance_100m"]
                now_g, old_g = row[f"{metric}_yoy_pp"] / 100, old[f"{metric}_yoy_pp"] / 100
                current = 100 * math.log(now_level / old_level)
                base = -100 * math.log((now_level / (1 + now_g)) / (old_level / (1 + old_g)))
                total = 100 * math.log((1 + now_g) / (1 + old_g))
                current_terms.append(current)
                base_terms.append(base)
                growth_terms.append(total)
                for name, value in [("current", current), ("base_revision", base), ("growth", total)]:
                    residual = abs(row[f"d{lag}_{metric}_{name}_log_pp"] - value)
                    maximum_residual = max(maximum_residual, residual)
                    assert residual < 1e-10
                    identities += 1
            for name, values in [("current", current_terms), ("base_revision", base_terms), ("growth", growth_terms)]:
                assert abs(row[f"d{lag}_relative_{name}_log_pp"] - (values[0] - values[1])) < 1e-10
                identities += 1
    label_columns = [col for col in frame if col.startswith(("E0_", "E1_"))]
    for col in label_columns:
        a = frame[col].reset_index(drop=True)
        b = parent.loc[frame.stat_month, col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a):
            assert np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True), col
        else:
            assert a.fillna("缺失").equals(b.fillna("缺失")), col
    summary = pd.read_csv(STUDY / "results/全部机制分组_描述分布.csv")
    for row in summary.to_dict("records"):
        part = frame[(frame.training_regime == row["training_regime"]) & (frame.arithmetic_mechanism_group == row["group"])]
        values = part[f"{row['entry']}_{row['horizon']}_return"].dropna()
        assert len(values) == row["mature_months"]
        assert len(part) == row["all_months"]
        if len(values):
            assert abs(values.mean() - row["return_mean"]) < 1e-12
            assert abs(values.median() - row["return_median"]) < 1e-12
    clocks = pd.read_csv(STUDY / "results/104个月_机制证据时钟.csv")
    assert clocks.cause_status.eq("NOT_IDENTIFIED").all()
    assert clocks.annual_table_historical_admission.eq("EXCLUDED_NO_ORIGINAL_MONTHLY_VINTAGE").all()
    late = frame[frame.stat_month == "2026-08"].iloc[0]
    assert pd.isna(late.E0_20_return) and pd.isna(late.observation_date)
    deposits = pd.read_csv(STUDY / "results/全国存款结构_2023至2025_事后快照.csv")
    assert len(deposits) == 36 and deposits.historical_decision_admission.eq("EXCLUDED").all()
    source_receipts = json.loads((STUDY / "sources/source_receipts.json").read_text(encoding="utf-8"))
    saved_sources = 0
    for source in source_receipts:
        if source.get("local_path"):
            assert digest(STUDY / source["local_path"]) == source["sha256"]
            saved_sources += 1
        else:
            assert source["id"] == "wealth_2024H1" and source["local_pdf_saved"] is False
            assert source["route_stopped_after_two_failures"]
    for document in ["第二轮机制观察结论.md", "00_阅读与复算.md"]:
        text = (STUDY / document).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target:
                assert (STUDY / target).exists(), target
    shutil.copy2(Path(__file__), STUDY / "code" / Path(__file__).name)
    results = {"status": "PASS_SAVED_OUTPUT_RECOMPUTATION", "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "monthly_rows": 104, "recomputed_arithmetic_cells": identities, "max_residual_log_pp": maximum_residual, "unchanged_parent_label_columns": len(label_columns), "group_summary_rows_recomputed": len(summary), "deposit_months": len(deposits), "new_local_sources_hash_checked": saved_sources, "online_only_source_count": 1, "figures_visually_reviewed": 2, "pdf_full_relevant_pages_visually_reviewed": ["pbc_2024Q3:printed28", "pbc_2024Q3:printed29"], "market_cutoff": "2026-09-11", "historical_vintage_proof": "NOT_PROVEN", "causes": "UNKNOWN_OR_MIXED", "positions": "NOT_COMPUTED", "sharpe": "NOT_COMPUTED", "strategy_validation": "NOT_ESTABLISHED", "zip_created": False}
    (STUDY / "verification_saved_outputs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = []
    for path in sorted(STUDY.rglob("*")):
        if path.is_file() and path.name != "completion_receipt.json":
            outputs.append({"path": str(path.relative_to(STUDY)), "bytes": path.stat().st_size, "sha256": digest(path)})
    receipt = {"status": "COMPLETED_DESCRIPTIVE_MECHANISM_ROUND", "completed_at": results["finished_at"], "protocol_sha256": digest(STUDY / "protocol.json"), "result": "完成104月余额/隐含基数分解、36月存款结构、机制时钟与原行情标签描述比较；未识别唯一月度因果或独立预测增量。", "known_limitations": ["旧分组季节结构不平衡", "新同向仅2个月", "隐含基数包含舍入/修订", "年度表不具原始逐月版本", "理财PDF仅在线读取，本地两次超时停止"], "files": outputs}
    (STUDY / "completion_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
