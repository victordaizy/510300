"""仅整理已保存具体解释的表格和阶段含义，不重跑定义或连接新标签。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.down_gap_reclaim_explanation_v1 import OUT
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    receipt = OUT / "explanation_delivery_receipt.json"
    require(not receipt.exists(), "解释交付已完成，不覆盖。")
    summary = read(OUT / "summary.json")
    require(summary["technical_decision"] == "TECH.R183" and summary["new_accounts"] == 0, "原解释未完成。")
    for item in read(OUT / "protocol.json")["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结原解释来源改变。")
    report = OUT / "具体上涨与点位反推.md"
    original = report.read_text(encoding="utf-8")
    backup = OUT / "initial_report_before_table_layout_fix.md"
    with backup.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(original)
    blocks = original.split("## 原案例")
    rebuilt = [blocks[0]]
    for block in blocks[1:]:
        rows = block.splitlines()
        table_positions = [i for i, row in enumerate(rows) if row.startswith("|")]
        if table_positions:
            first = min(table_positions)
            all_table_rows = [rows[i] for i in table_positions]
            rest = [row for i, row in enumerate(rows) if i not in table_positions]
            rest[first:first] = all_table_rows + [""]
            block = "\n".join(rest)+"\n"
        rebuilt.append("## 原案例"+block)
    text = "".join(rebuilt)
    explanation = (
        "## 四段的阶段区别与下一可知点位\n\n"
        "2019春季是先形成上升路径，再在3月8日回调产生完整下跌缺口、3月18日收复。"
        "收复时日DIF和上一完整周柱已正、日柱负、相对量0.975；它描述趋势中的回调修复，"
        "不能解释为提前捕捉1月低点，也不能要求放量才算已有收复。\n\n"
        "2020年3月低点后，4月2日日柱先正而DIF和上一完整周柱仍负，"
        "相对量0.741、RV比1.781；它描述下跌后的局部修复，长周期指标仍反映此前跌势。"
        "6月1日收复另一锚时DIF小幅正、日柱和上一周柱仍负，相对量1.201。"
        "后来的7月上涨不能单凭这些符号预先保证，必须看按失效规则实际持有到了何时。\n\n"
        "2024年9月24日收盘3.427已越过9月9日原缺口上沿3.300，"
        "量是前20日中位量的3.338倍、日柱正而DIF/上一周柱负；"
        "它在原5%确认同日提供了可知的区间收复信息，早于这两项较慢动量转正。"
        "9月25日真实开盘才可尝试，不能按9月13日事后低点成交；"
        "量和MACD的组合仅作解释，不能看完这一个赢家才加入过滤。\n\n"
        "2015年7月9日收复出现在原6月29—30短反弹之后，日周动量仍负、RV比3.620。"
        "此前多次新的向下缺口替换旧锚，显示价格向下区间分离反复发生；"
        "单次收复只否定最新一个区间，不能证明整个下行已结束。\n\n"
    )
    text = text.replace("## 全体与可检验点位", explanation+"## 全体与可检验点位")
    report.write_text(text, encoding="utf-8", newline="\n")
    write_json(receipt, {
        "at": now(), "status": "PASS_FOUR_CASE_CHARTS_AND_SAVED_STAGE_EXPLANATION_DELIVERED",
        "charts_visually_inspected": summary["charts"], "table_layout_corrected_from_saved_rows": True,
        "definition_or_policy_changes": 0, "new_account_runs": 0, "new_result_label_joins": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), OUT / "summary.json", backup, report]],
    }, exclusive=True)
    print("四图已查看，全部收复表格与四段阶段解释已交付；无新账户或规则修改。", flush=True)


if __name__ == "__main__":
    run()
