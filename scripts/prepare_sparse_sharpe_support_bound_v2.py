"""依据用户明确指令更新净夏普为1.3，保存原目标和原分析。"""
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_sharpe_support_bound_v2"
OLD = ROOT / "reports/research/510300_sparse_sharpe_support_bound_v1"
MANDATE = ROOT / "config/510300_existing_data_training_mandate_v1.json"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if OUT.exists():
        raise ValueError("本轮目录已存在，不覆盖。")
    original = json.loads(MANDATE.read_text(encoding="utf-8"))
    if original["target_net_sharpe"] != 1.5:
        raise ValueError("原目标已变化，请核对用户指令。")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    history = OUT / "authority_update"
    history.mkdir(parents=True)
    shutil.copy2(MANDATE, history / "previous_mandate.json")
    for name in ["protocol.json", "summary.json", "formula_checks.json"]:
        shutil.copy2(OLD / name, history / ("previous_1_5_" + name))
    updated = dict(original)
    updated.update({"latest_user_instruction": "夏普1.3就行", "target_net_sharpe": 1.3,
                    "target_revision_at": stamp, "previous_target_net_sharpe": 1.5,
                    "earlier_user_instruction_still_active": "不用采集；直接使用已有本地数据训练",
                    "target_revision_receipt": "reports/research/510300_sparse_sharpe_support_bound_v2/authority_update/user_target_revision.json"})
    save(history / "user_target_revision.json", {"received_instruction": "夏普1.3就行", "recorded_at": stamp,
        "previous_target_net_sharpe": 1.5, "current_target_net_sharpe": 1.3,
        "capital_cny": 200000, "target_max_drawdown": 0.1,
        "executable_assets": ["510300.SH", "CASH_CNY"], "new_collection_enabled": False,
        "historical_results_modified": False, "orders_authorized": False})
    protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))
    protocol.update({"study_id": "510300_SPARSE_SHARPE_SUPPORT_BOUND_V2",
        "user_instruction": "只做510300，20万元，用户最新改为净夏普至少1.3，最大回撤10%；低频参与，不用采集。",
        "target_net_sharpe": 1.3, "previous_target_net_sharpe": 1.5,
        "revision_reason": "用户明确说夏普1.3就行；重新计算目标判定，保留V1及各来源原始结果。",
        "numeric_target_reassessment": "从原净值重新计算夏普和回撤，再与1.3及10%比较；不得沿用来源旧目标的通过标记。"})
    save(OUT / "protocol.json", protocol)
    save(MANDATE, updated)
    print("用户目标已更新为净夏普1.3；20万元、最大回撤10%、资产范围与不采集指令保持。")


if __name__ == "__main__":
    main()
