"""保存已复算并完成图表目视检查的本轮交付记录。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_bank_profit_credit_bridge_v13"


def main():
    target = OUT / "completion_receipt.json"
    if target.exists():
        raise RuntimeError("本轮完成记录已存在，不覆盖。")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    publication = json.loads((OUT / "publication_receipt.json").read_text(encoding="utf-8"))
    report = OUT / result["report"]
    figure = OUT / result["figure"]
    assert sha256(report.read_bytes()).hexdigest() == publication["report_sha256"]
    assert sha256(figure.read_bytes()).hexdigest() == publication["figure_sha256"]
    facts = pd.read_csv(OUT / "results/原表金额与期间来源.csv")
    assert len(facts) == 46 and facts[["company", "key"]].duplicated().sum() == 0
    assert pd.to_datetime(facts.known_at, utc=True).max() < pd.Timestamp(result["snapshot_at"])
    p = facts[facts.company.eq("平安银行")].set_index("key")
    for col in ["previous", "current"]:
        np.testing.assert_allclose(p.loc["interest_income", col] - p.loc["interest_expense", col], p.loc["net_interest", col])
        np.testing.assert_allclose(p.loc["credit_impairment", col] + p.loc["other_impairment", col], p.loc["total_impairment", col])
        net = p.loc["revenue", col] - p.loc["opex", col] - p.loc["tax_surcharge", col] - p.loc["total_impairment", col]
        net += p.loc["nonop_income", col] - p.loc["nonop_expense", col] - p.loc["income_tax", col]
        np.testing.assert_allclose(net, p.loc["parent_profit", col])
        ordinary = net - p.loc["preferred_dividend", col] - p.loc["perpetual_interest", col]
        np.testing.assert_allclose(ordinary, p.loc["ordinary_profit", col])
        np.testing.assert_allclose(np.round(ordinary / p.loc["weighted_shares", col], 2), p.loc["eps", col])
    bridge = pd.read_csv(OUT / "results/平安银行_净利润增量金额桥.csv")
    nii = pd.read_csv(OUT / "results/平安银行_净利息收入规模与综合比率分解.csv")
    np.testing.assert_allclose(bridge.profit_change_yi.sum(), (p.loc["parent_profit", "current"] - p.loc["parent_profit", "previous"]) / 100)
    np.testing.assert_allclose(nii.change_yi.sum(), (p.loc["net_interest", "current"] - p.loc["net_interest", "previous"]) / 100)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    artifacts = {}
    for name in [result["report"], result["figure"], "result.json", "verification.json", "protocol.json", "freeze.json",
                 "results/原表金额与期间来源.csv", "results/平安银行_净利润增量金额桥.csv", "results/平安银行_净利息收入规模与综合比率分解.csv"]:
        artifacts[name] = sha256((OUT / name).read_bytes()).hexdigest()
    record = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_PROFIT_AND_EPS_RECOMPUTATION",
        "study_status": result["status"], "continuation_classification": "PROGRESS", "goal_achieved": False,
        "independent_validation": False, "new_models": 0, "new_accounts": 0,
        "visual_check": {"status": "PASS_VISUAL_INSPECTION", "path": result["figure"], "sha256": artifacts[result["figure"]],
                         "notes": "图中金额与保存表一致，中文标题及脚注完整，坐标轴从零起，无文字遮挡或裁切。"},
        "checks": ["46项原表事实键唯一", "原公告保守可用日在观察点之前", "收入至净利润金额一致", "信用与其他资产减值相加一致", "普通股利润与EPS一致", "已保存的两组增量分解一致"],
        "artifacts_sha256": artifacts}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"交付状态": record["status"], "本轮研究": record["study_status"], "目标完成": False, "复核事实": len(facts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
