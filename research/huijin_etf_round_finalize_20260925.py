"""将汇金公开披露研究的已保存结果写成中文结论，并更新主线状态。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_huijin_etf_event_saved_completion_v1"
SOURCE = ROOT / "reports/research/510300_huijin_etf_disclosure_completion_v1"
PARENT = ROOT / "reports/research/510300_huijin_etf_event_probe_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run():
    result = read(OUT / "result.json")
    source = read(SOURCE / "admitted_source_result.json")
    primary = result["primary"]
    control = next(row for row in result["all_accounts"] if row["policy"] == "DISCLOSURE_ONLY" and row["cost"] == "STRESS")
    comparison = next(row for row in result["comparisons"] if row["cost"] == "STRESS")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    account_lines = []
    for name, label in [("DISCLOSURE_ONLY", "公告后参与"), ("DISCLOSURE_PRICE_CONFIRMATION", "公告后等待价格确认")]:
        cycles = pd.read_parquet(PARENT / "accounts/STRESS" / name / "cycles.parquet")
        account_lines.append(f"{label}的逐笔记录保存在原账户目录，共{len(cycles)}个完整持仓周期。")
    report = f"""本轮没有找到合格的高夏普策略。新增取得汇金官网当前15个年度目录的114篇文章，完成8篇ETF相关披露的语义分类，并实际运行两条固定规则、两档成本共4个完整账户。主方案在压力成本下夏普{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，20万元账户期末{primary['end_equity']:.2f}元。高夏普总目标保持进行中。

用户当前要求保持：仅510300与现金，20万元，压力成本净夏普至少1.2、复合年化至少10%、最大回撤不超过10%；只用最近两个日历年的训练信息，每天更新；单笔3:1已改为账户尾部预算。该预算并不保证实际跳空和T+1约束下的最终回撤。

新证据的范围是汇金官网当前可见2012—2026年度目录，共114篇文章。8篇ETF相关披露中，6篇涉及已经发生的买入，1篇披露出售，1篇只有未来增持承诺。2013年的买入保留为来源事实；账户评价从2015年开始，2015年7月的两次买入披露依固定规则合并、不延期，最终账户时期只有4个独立买入事件。不能把114篇文章、8篇披露或重复日样本当成114个独立交易事件。

本次补齐的是当前官网目录，不代表全国所有政策事件，也没有认证每篇网页的历史首版。文章中没有披露精确510300买入份额及金额，不能据此重建510300每日资金流。2015年5月出售披露和2025年4月只有承诺的文章均保留，未只收集增持成功案例。

两条规则在看事件收益前固定。公告时间只有日期时，统一用当日23:59:59作为可用上界；首个完整交易日收盘才评估，次开盘最早成交。对照方案在评估后参与；主方案等待含分红价格高于公告前最后交易日的参考价后参与。成交后价格失效，或固定20日窗口到期，下一可卖开盘退出；同一事件退出后不再进入，重复披露不延长窗口。两者采用相同的两年滚动尾部估计、仓位预算、T+1、分红、整手和费用模型。

完整账户覆盖2015-01-05至2026-09-24的2852个交易日，包括所有空仓日；没有在不同年份重新注资。压力成本结果如下：

- 公告后参与：夏普{control['sharpe']:.6f}，复合年化{control['annual_return']:.4%}，最大回撤{abs(control['max_drawdown']):.4%}，净损益{control['profit']:.2f}元，4个完整周期。
- 公告后等待价格确认：夏普{primary['sharpe']:.6f}，复合年化{primary['annual_return']:.4%}，最大回撤{abs(primary['max_drawdown']):.4%}，净损益{primary['profit']:.2f}元，3个完整周期。

2015-07-07买入至07-08退出，两条规则都亏6248.20元，约为初始账户的3.12%。这直接说明已披露机构购买不排除后续继续下跌。2023年等待确认将对照的824.52元亏损变为148.13元盈利，但2025年主方案未入场，错过对照2342.00元盈利。2024年的两账户盈利都保留；因前序权益和风险预算不同，金额差不能当作相同仓位的纯信号增量。

主方案相对对照的压力成本算术年化差为{comparison['annual_arithmetic_difference']*100:.5f}个百分点，20日区块、2000次重采样的95%区间为[{comparison['lower_95']*100:.5f}, {comparison['upper_95']*100:.5f}]个百分点；未做整个项目的选择校正。4个完整账户均未达到数值门槛，主压力账户的滚动两年联合通过次数为0。此结果拒绝当前固定用途，不等于证明所有政策信息永久没有价值，也不据此更换窗口挽救失败。

已完成2853份每日两年尾部估计，其中2846份与已有共同来源逐值一致。账户现金、含分红权益、风险约束和次开盘顺序核对通过；截至2024年底的2431日账户及全部决策，与截断未来数据的重放精确一致。初次复核因datetime存储精度不同中断，原失败记录保留；完成步骤只统一比较时的日期存储单位，没有改变日期值、策略、费用或四个账户。

本轮新完整账户4个、收益模型拟合0个、参数扫描0次、独立前向观察0个。历史重建结果仍是开发证据。原PCF/IOPV自动采集保持暂停，没有订单执行，没有制作审核包或用户表格。

{chr(10).join(account_lines)}

关键文件：本目录result.json为最终结果，原四账户在../510300_huijin_etf_event_probe_v1/accounts，原协议在该父目录protocol.json；完整来源结果在../510300_huijin_etf_disclosure_completion_v1/admitted_source_result.json，8篇事实在同目录classified_etf_facts.json。

官方原文：[2015年出售披露](https://www.huijin-inv.cn/huijin-inv/Corporate_History/2015-05/1000740.shtml)、[2015年7月买入披露](https://www.huijin-inv.cn/huijin-inv/Corporate_History/2015-07/1000738.shtml)、[2023年买入披露](https://www.huijin-inv.cn/huijin-inv/c100074/2023-10/1002226.shtml)、[2024年扩大增持披露](https://www.huijin-inv.cn/huijin-inv/c100077/2024-02/1002262.shtml)、[2025年再次增持](https://www.huijin-inv.cn/huijin-inv/SC20252/2025-04/1002841.shtml)。
"""
    (OUT / "汇金新证据与账户结论.md").write_text(report, encoding="utf-8")
    completed = {"at": now, "status": "NEW_OFFICIAL_DISCLOSURE_EVIDENCE_AND_FOUR_ACCOUNTS_COMPLETED_TARGET_UNMET",
                 "studies": ["510300_HUIJIN_ETF_DISCLOSURE_SOURCE_V1", "510300_HUIJIN_ETF_DISCLOSURE_COMPLETION_V1",
                             "510300_HUIJIN_ETF_EVENT_PROBE_V1", "510300_HUIJIN_ETF_EVENT_SAVED_COMPLETION_V1"],
                 "source_documents": source["documents"], "catalog_year_count": len(source["catalog_years"]),
                 "etf_disclosures": source["etf_documents"], "account_period_buy_episodes": 4,
                 "new_full_accounts": 4, "daily_tail_estimates": 2853, "saved_tail_estimates_reproduced": 2846,
                 "technical_prefix_replays": 2, "technical_replays_are_new_candidates": False,
                 "new_return_model_fits": 0, "parameter_searches": 0, "independent_forward_observations": 0,
                 "primary": primary, "control_stress": control, "goal_achieved": False,
                 "goal_status": "active", "orders_authorized": False, "review_package_created": False}
    write(OUT / "completed_round.json", completed)
    status_path = MAIN / "current_status.json"
    status = read(status_path)
    for name in completed["studies"]:
        if name not in status["completed_followup_studies"]:
            status["completed_followup_studies"].append(name)
    relative = OUT.relative_to(ROOT).as_posix()
    status.update(updated_at=now, goal_achieved=False, goal_status="active",
                  latest_goal_turn_classification="PROGRESS_NEW_OFFICIAL_EVENT_CORPUS_AND_FOUR_FIXED_ACCOUNTS",
                  same_condition_consecutive_no_progress_goal_turns=0,
                  active_blocker_id=None, active_blocker_description=None,
                  latest_user_requested_study=relative,
                  latest_user_requested_study_status=completed["status"], research_execution_state=completed["status"],
                  latest_continuation_report=relative + "/汇金新证据与账户结论.md",
                  latest_huijin_disclosure_round=completed,
                  goal_metadata_note="goal工具为active；新增官方目录证据与四个完整账户已完成，目标仍未完成。",
                  remaining_research_question="当前两年日更与尾部预算下仍缺足够的可执行收益优势及独立验证；本次固定政策披露用途未通过。")
    write(status_path, status)
    (MAIN / "最新研究结论.md").write_text(report + "\n全部此前研究的数值、失败和授权变化保留在current_status.json及各原研究目录。\n", encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(latest_progress_receipt=relative + "/completed_round.json",
                   last_research_result="补齐汇金当前官网114篇文章，完成四个固定账户；政策披露与价格确认均未达标，高夏普目标保持active。",
                   latest_integrated_experiment=result["study_id"],
                   latest_continuation_report=relative + "/汇金新证据与账户结论.md",
                   latest_continuation_classification=status["latest_goal_turn_classification"],
                   research_execution_state=completed["status"], goal_status="active", goal_achieved=False)
    write(mandate_path, mandate)
    status_md = ROOT / "RESEARCH_STATUS.md"
    marker = "2026-09-25 汇金官方披露母集与四账户完成"
    previous = status_md.read_text(encoding="utf-8-sig")
    if marker not in previous:
        banner = (f"> {marker}：当前官网15年度、114篇文章，8篇ETF披露，账户时期4个独立买入事件。"
                  f"公告参与/价格确认压力夏普{control['sharpe']:.3f}/{primary['sharpe']:.3f}，均未达标；"
                  "新增收益拟合0、独立前向0。当前目标保持ACTIVE，旧结果不变，不制作审核包和用户表格。"
                  f"见[研究结论]({relative}/汇金新证据与账户结论.md)。\n\n")
        status_md.write_text(banner + previous, encoding="utf-8")
    print("汇金来源和四账户结论已保存；主线状态已同步为进行中、未达标。", flush=True)


if __name__ == "__main__":
    run()
