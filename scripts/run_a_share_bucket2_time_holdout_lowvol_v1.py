"""一次性打开桶2未见未来收益上的单因子低波动结果。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_a_share_bucket1_time_holdout_formula_v1 as implementation  # noqa: E402
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS  # noqa: E402
from scripts.freeze_a_share_bucket2_time_holdout_lowvol_v1 import verify_protocol  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402


def select_targets(features: pd.DataFrame, contract: dict) -> pd.DataFrame:
    ready = features.loc[
        features["signal_output"].eq("SIGNAL_READY")
        & features["raw_close"].ge(float(contract["universe"]["minimum_signal_price_cny"]))
    ].copy()
    ready["score"] = ready[FEATURE_COLUMNS[4]]
    ready.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
    selected = ready.groupby("date", sort=True).head(1).copy()
    selected["selection_rank"] = 1
    targets = selected[["date", "con_code", "score", "selection_rank"]].rename(columns={"date": "signal_date"})
    targets["regime"] = "BUCKET2_TIME_HOLDOUT_LOWVOL20"
    return targets


def render(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# 全A桶2代码与时间盲测低波动公式 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 因子数：1",
        "- 桶2未来收益在公式冻结后下载",
        "",
        "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交额：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += ["", "结果不生成仓位、订单或券商连接。", ""]
    return "\n".join(lines)


def main() -> int:
    implementation.verify_protocol = verify_protocol
    implementation.select_targets = select_targets
    implementation.render = render
    result = implementation.main()
    contract, _ = verify_protocol()
    result_path = ROOT / contract["paths"]["result_json"]
    report = json.loads(result_path.read_text(encoding="utf-8"))
    report["status"] = (
        "STRICT_BUCKET2_CODE_AND_TIME_HOLDOUT_PASS_AWAITING_FORWARD"
        if all(report["gates"].values())
        else "STRICT_BUCKET2_CODE_AND_TIME_HOLDOUT_REJECTED_FROZEN"
    )
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), result_path)
    atomic_text(render(report), ROOT / contract["paths"]["result_markdown"])
    print(json.dumps({
        "桶2最终状态": report["status"],
        "基础年化超额": report["base_cost"]["annualized_excess"],
        "压力年化超额": report["stress_cost"]["annualized_excess"],
    }, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
