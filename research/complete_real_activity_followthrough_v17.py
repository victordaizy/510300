"""记录第十七轮已完成的传导诊断，以及仍需解决的研究目标。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_real_activity_followthrough_v17"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮已完成，不覆盖完成记录。")
    verify = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verify["status"] == "PASS_SAVED_REAL_ACTIVITY_CLOCK_PATH_RECOMPUTATION"
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    selected = pd.read_csv(OUT / "results/原11个共同支持月_全部后续证据.csv")
    assert len(selected) == 11
    assert (selected.orders_next1_value > 50).all()
    assert (selected.orders_next3_mean_minus_origin > 0).sum() == 5
    observed = selected[["orders_next1_value", "orders_next2_value", "orders_next3_value"]].ge(50).all(axis=1)
    assert observed.sum() == 7
    assert abs(selected.E0_return_percent.mean() - .1508858751528184) < 1e-8
    assert (selected.confirmation_fresh_entry_return_percent > 0).sum() == 5
    at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    completion = {"at": at, "study_id": "510300_REAL_ACTIVITY_FOLLOWTHROUGH_V17", "status": verify["status"],
                  "continuation_classification": "PROGRESS", "goal_achieved": False, "independent_validation": False,
                  "checks": verify, "frozen_protocol_unchanged": True,
                  "visual_check": {"status": "PASS_VISUAL_INSPECTION", "notes": "已查看最终图，标题与表头无重叠；11个月完整显示，后续经济列与原E0/E1收益分开。"},
                  "report_sha256": digest(OUT / "第十七轮_经营传导兑现与剩余收益.md"),
                  "figure_sha256": digest(OUT / "figures/订单延续与股价路径_全部11个月.png"),
                  "complete_104_month_table_sha256": digest(OUT / "results/104个月_后续经营结果与确认时钟.csv"),
                  "limitation": "经营延续、回款变化和剩余收益的历史核对；不能据此识别因果份额、预期意外或稳定交易优势。"}
    progress = {"at": at, "classification": "PROGRESS", "overall_goal_status": "active", "goal_achieved": False,
                "new_evidence": ["原11个共同支持月下一期订单全部高于50；7个月后三期均不低于50，但仅5个月三期均值高于起点。",
                                 "2021年1月后续订单及部分回款指标改善，原20日仍跌11.09%；确认下一期订单后至原终点仍跌5.72%。",
                                 "等待下一期订单的11个月有5个月剩余收益为正，均值1.56%、中位数-0.14%，正反路径全部保留。"],
                "next_bounded_question": "以2020年11月至2021年1月连续三个既有月份为下一段机制观察，复用已核对的货币调查预期并补对应利率消息时钟，区分经营状态、公告相对预期及权重行业价格变化；不得把这三个已知结果当独立验证，不重训旧失败模型。",
                "unresolved": ["同一经营状态下真正的新信息与市场预期", "盈利预期、折现率与风险溢价的相对作用", "跨时期独立样本中的可利用增量"],
                "new_models": 0, "new_accounts": 0, "orders_authorized": False}
    for name, obj in [("completion_receipt.json", completion), ("goal_progress.json", progress)]:
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name in [Path(__file__).name, "publish_real_activity_followthrough_v17.py", "real_activity_followthrough_v17.py"]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    print("第十七轮完成：104个月、700条后续披露与全部确认路径已落盘。")
    print("总目标继续进行；经营状态延续尚未转化为经过独立验证的510300确定性方向。")


if __name__ == "__main__":
    main()
