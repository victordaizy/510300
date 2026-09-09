"""只核对下一条件组合的既有来源和时点，不生成条件、目标或账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.downside_reference_risk_v1 import ROOT, OUT as P132, CONFIG as C132, PRIMARY
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    destination = ROOT / "reports/research/510300_downside_balance_gate_preflight_20260909"
    cfg = json.loads(C132.read_text(encoding="utf-8"))
    proposal = ROOT / "docs/510300_DOWNSIDE_BALANCE_GATE_NEXT_20260909.md"
    paths = [C132, proposal, P132 / "saved_verification_receipt.json", P132 / "risk_factors.parquet"]
    risk = pd.read_parquet(P132 / "risk_factors.parquet")
    coverage = []
    for period, start, end in [("evaluation", cfg["evaluation_start"], cfg["data_cutoff"]),
        ("earlier_diagnostic", cfg["earlier_start"], cfg["earlier_terminal"])]:
        expected = risk[risk.date.le(end)]
        first = int(np.flatnonzero(expected.date.ge(start))[0])
        indices = np.arange(first-1, len(expected)-1)
        for cost in cfg["costs"]:
            path = P132 / period / cost / f"{PRIMARY}_factors.parquet"
            decision_path = P132 / period / cost / f"{PRIMARY}_decisions.parquet"
            paths.extend([path, decision_path])
            factors = pd.read_parquet(path)
            decisions = pd.read_parquet(decision_path)
            require(pd.DatetimeIndex(factors.date).equals(pd.DatetimeIndex(expected.date)), "下一项父目标日历不同")
            require(factors.reference_cost.eq(cost).all() and factors.model.eq(PRIMARY).all(), "下一项父目标费用或设置错误")
            require(np.array_equal(factors.origin_index, np.arange(len(expected))), "下一项父目标索引被截断")
            require(np.array_equal(decisions.origin_index, indices), "下一项父判断不是完整已知收盘")
            require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(expected.date.iloc[indices])), "下一项父目标收盘时钟不同")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(expected.date.iloc[indices+1])), "下一项父目标不是下一开盘")
            np.testing.assert_allclose(decisions.reference_weight, factors.target.iloc[indices], atol=1e-12, rtol=0, equal_nan=True)
            require(factors.target.iloc[:first-1].isna().all() and pd.isna(factors.target.iloc[-1]), "下一项把无参考准备期或末日收盘当成已知")
            for column in ["full_second_moment20", "downside_second_moment20"]:
                np.testing.assert_allclose(factors[column], expected[column], atol=1e-15, rtol=0, equal_nan=True)
            target = factors.target.iloc[indices].to_numpy(float)
            values = factors.iloc[indices][["full_second_moment20", "downside_second_moment20"]].to_numpy(float)
            require(np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all(), "下一项保存父目标缺失或超出不融资范围")
            require(np.isfinite(values).all() and (values >= 0).all(), "下一项评价中的平方风险来源不足")
            require((values[:, 1] <= values[:, 0]+1e-15).all(), "下行平方幅度超过全部平方幅度")
            coverage.append({"period": period, "cost": cost, "calendar_rows": len(factors), "known_origin_rows": len(indices),
                "first_origin": str(decisions.origin.iloc[0].date()), "last_origin": str(decisions.origin.iloc[-1].date()),
                "matched_next_open_clock": True, "complete_risk_inputs_at_all_origins": True})
    result = {"checked_at": now(), "status": "EXISTING_PARENT_TARGETS_COST_CLOCKS_AND_SQUARED_RISK_INPUTS_AVAILABLE",
        "proposed_round": 133, "registered": False, "coverage": coverage,
        "new_gates_computed": 0, "new_targets_computed": 0, "new_models_fit": 0, "new_accounts_generated": 0,
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths], "reviewer_source_sha256": digest(Path(__file__))}
    write_json(destination / "result.json", result, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 132, "下一项预检不能覆盖已经变化的研究索引")
    index["next_work"].update(status="DOWNSIDE_BALANCE_GATE_INPUT_CLOCK_PREFLIGHT_COMPLETE", preflight=str((destination / "result.json").relative_to(ROOT)), registered=False)
    write_json(index_path, index)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
