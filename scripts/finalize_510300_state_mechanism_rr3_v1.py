"""从保存结果拆分准入原因，更新中文结论；不重跑或调整冻结规则。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_state_mechanism_rr3_v1"
MAIN = WORKSPACE / "reports/research/510300_integrated_research_continuation_20260924"
V3 = WORKSPACE / "reports/research/510300_pattern_daily_state_learning_v3_saved_completion"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    if (ROOT / "saved_diagnostic.json").exists():
        raise RuntimeError("本轮保存诊断已完成；不得重新选择口径。")
    out = ROOT / "results"
    x = pd.read_parquet(out / "known_states.parquet")
    candidates = pd.read_parquet(out / "all_candidate_facts.parquet")
    candidates = candidates[candidates.entry_date.between("2021-01-04", "2026-08-14")].copy()
    ready = candidates.reference_ticket_status.eq("READY")
    price = ((candidates.kind == "TREND_PULLBACK") & (candidates.market_state == "UP")) | ((candidates.kind == "SHOCK_REPAIR") & candidates.market_state.isin(["RANGE", "DOWN"]))
    known = candidates.state_information_known
    mechanism = np.where(candidates.kind == "TREND_PULLBACK", candidates.trend_mechanism_support, candidates.repair_mechanism_support)
    funnel = {"all_price_candidates": int(len(candidates)), "reference_rr3": int(ready.sum()),
              "rr3_by_strategy": {str(k): int(v) for k, v in candidates.loc[ready].kind.value_counts().items()},
              "rr3_and_price_route": int((ready & price).sum()), "rr3_price_route_and_common_information": int((ready & price & known).sum()),
              "rr3_price_route_common_information_and_economic_conditions": int((ready & price & known & mechanism).sum())}
    detail = []
    for row in candidates.loc[ready].itertuples():
        fact = x.iloc[row.entry_idx]
        stages = {"price_state_pass": bool(price.loc[row.Index]), "information_known": bool(row.state_information_known),
                  "economic_condition_pass": bool(mechanism[candidates.index.get_loc(row.Index)])}
        detail.append({"signal_id": row.signal_id, "entry_date": row.entry_date, "strategy": row.kind,
                       "market_state": row.market_state, "macro_state": row.macro_state, "funding_state": row.funding_state,
                       "reference_net_rr": float(row.reference_planned_net_rr),
                       "known_new_orders": float(fact.pmi_level + 50), "new_orders_change3": float(fact.pmi_change3),
                       **stages})
    # 保留原冻结输出，另存语义注释，零路径的机械区间不能解释为真实增量区间。
    result = load(ROOT / "result.json")
    primary = result["primary_accounts"]
    matched = load(out / "accounts/STRESS/PRICE_ROUTE_MATCHED/metrics.json")
    all_cash = all(r["completed_cycles"] == 0 and r["average_exposure"] == 0 for r in primary) and matched["average_exposure"] == 0
    diagnostic = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "保存结果的原因拆分，不改变冻结规则，不新建账户。",
        "funnel": funnel, "rr3_candidate_facts": detail,
        "inference_status": "NOT_IDENTIFIABLE_BOTH_PRIMARY_AND_MATCHED_ALL_CASH" if all_cash else "SEE_FROZEN_PAIRED_RESULT",
        "mechanical_bootstrap_interval_is_evidence_interval": False if all_cash else None,
        "formal_economic_increment_interval": None if all_cash else result["paired_increment"]["ci95"],
        "reason": "两条路径均完全空仓，保存的[0,0]只是零数组的机械重抽结果；不证明宏观无效，也不证明风险为零。" if all_cash else None,
        "rule_compatibility_finding": "7个满足3:1的趋势回调均在120日均线上方，但20日标准化动量状态为RANGE；本版同时要求短期UP，把这些回调排除。",
        "new_fits": 0, "new_accounts": 0,
        "original_results_preserved": True,
        "json_boolean_note": "原冻结序列化器将Python布尔写为0/1；以下注释使用标准布尔。仅工作源修正了序列化类型，未修改冻结副本或重跑。",
    }
    # 核对输入时钟及冻结表示的前缀因果性，不重算任何账户。
    spec = importlib.util.spec_from_file_location("frozen_rr3_diagnostic", ROOT / "code/state_mechanism_rr3_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d, dividends, information = module.load(ROOT)
    state_checks = 0
    for cutoff in ("2022-06-30", "2024-12-31", "2025-12-31"):
        stop = int(d.index[d.date.le(cutoff)][-1]) + 1
        xx, cc = module.prepare(d.iloc[:stop], information.iloc[:stop])
        fields = ["market_state", "flow_change3", "breadth_change3", "state_information_known", "trend_mechanism_support", "repair_mechanism_support"]
        pd.testing.assert_frame_equal(xx[fields], x.iloc[:stop][fields])
        saved = pd.read_parquet(out / "all_candidate_facts.parquet")
        expected = saved[saved.entry_idx.lt(stop)].reset_index(drop=True)
        pd.testing.assert_frame_equal(cc.reset_index(drop=True), expected)
        state_checks += len(xx)
    time_checks = 0
    decisions = pd.to_datetime(x.decision_time)
    for known_field, available_field in [("funding_known", "dr_available_at"), ("funding_known", "policy_available_at"), ("flow_known", "flow_available_at"), ("pmi_known", "pmi_available_at")]:
        mask = x[known_field].fillna(False)
        available = pd.to_datetime(x.loc[mask, available_field])
        assert (available <= decisions.loc[mask]).all()
        time_checks += int(mask.sum())
    diagnostic["truncated_input_state_rows_checked_including_overlap"] = state_checks
    diagnostic["available_clock_field_values_checked"] = time_checks
    frozen = load(ROOT / "freeze.json")
    diagnostic["frozen_bytes_unchanged"] = all(hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in frozen["files"])
    assert diagnostic["frozen_bytes_unchanged"]
    save(ROOT / "saved_diagnostic.json", diagnostic)
    report = """已按用户确认的新要求运行：不同状态选择不同策略；每次计划净收益空间至少是计划损失的3倍，按压力成本计算；目标、失效与解释必须在入场前确定。本轮仍未得到可操作的高夏普策略。完整联合规则没有成交，夏普未定义，不能把空仓和零回撤算作成功。

这次与前几轮的区别是，两类策略本身有不同规则。趋势回调复用旧第23轮的长期均线上方、重新站上二十日均线触发，目标锚定此前六十日最高收盘，失效为已形成的五日低点下方，最多二十日。急跌修复复用旧五日急跌后的首日回升，目标锚定急跌前收盘，失效为该段已形成低点下方，最多五日。目标和止损按固定算法生成，不能因3:1不足而上移目标或缩窄失效范围。

主比较在价格状态之外增加已公布PMI新订单、DR007相对政策率及变化、沪深300成分股成交分类的支持或改善。趋势分支要求增长与股票参与支持，修复分支允许新订单仍低于50，但要求改善、资金利率不再上升、成交分类承压有所缓和。这是待检验的经济逻辑；PMI不是增长意外，成交分类不是具名机构资金流，历史高点也不是公平价值证明。

账户仍为20万元，只使用510300与现金，2021-01-04至2026-08-14共1361个交易日，252日年化。本轮在结果前固定单笔计划风险不超过账户1%、100份整手、T+1及两档成本。买价上限和份额在开盘前确定；开盘跳高使压力比例不足3，则不成交。价格目标或失效在收盘触发、下一可卖开盘执行，因而真实损失可以超过计划值。

94个候选包含44个趋势回调和50个急跌修复。按统一20万元参考票据计算，只有9个满足3:1：7个趋势回调、2个急跌修复。83个比例不足，另2个没有净上涨空间。3:1保持不变，没有为了增加交易降低要求。

问题出在后续路由的可用性。7个高比例趋势回调虽然都高于120日均线，但在本版20日标准化动量上全是RANGE；本版却只在短期UP时允许趋势回调，因此全部被价格路由排除。这揭示了长期背景与短期回调触发的冲突，不能把它解释为趋势回调在经济上不存在。2个高比例修复都符合价格路由，但都因次日开盘超出固定上限未成交；其中2021-07-29又缺共同资金资料，2021-12-22当时新订单处于收缩且走弱，经济支持条件也未通过。

只看价格路由、相同资料覆盖的价格路由、完整宏观资金路由，均无实际持仓。它们的差值是零，重抽也机械得到[0,0]；该区间不构成可识别的经济增量证据。本轮的宏观资金增量应标为NOT_IDENTIFIABLE_BOTH_PRIMARY_AND_MATCHED_ALL_CASH，而不是“证明宏观没有价值”。

不加外层状态选择的单趋势及两策略并集对照，在相同3:1与风险预算下实际完成4笔，全部来自趋势回调，压力净利润-6784.93元、年化-0.64%、夏普-0.455、最大回撤4.67%。4笔中1盈3亏，实际平均盈利／平均亏损只有0.109。3笔结构失效退出，1笔期限到期退出，没有一笔按目标退出。个别实际亏损超过计划损失，最差约-1.47R；2025-01-27节前那笔在二十日到期时仅盈利256.83元。这说明漂亮的计划比例不会自动兑现，也不能据此认定假期交易都不该做。

必须分清三种结论：本轮固定路由没有形成可检验的联合交易样本；两套价格目标的3:1本身没有建立收益优势；“不同状态应当使用不同策略”的一般假设尚未被这一轮证明或否定。尤其不能用一段短期动量同时定义市场大背景和回调买点，再把零交易当成经济结论。

后续仍遵守至少3:1与事前事实解释。研究重点需要区分背景、触发与失效，检验为什么某个价格目标有兑现依据，以及状态是否改变相对策略优势。本版窗口、阈值、目标和方向全部保留为冻结结果，不就地改成能挑中盈利交易的版本。历史样本均已被研究，独立前向观察仍为零。

本轮完成16条完整账户，其中包含现金及买入持有对照；没有新增模型拟合、数据下载或审核包。冻结字节、T+1、账户恒等式和已成交票据3:1均核对通过，另检查历史截断与来源可用时钟。这些实现检查不证明策略有效。
"""
    (ROOT / "研究结论.md").write_text(report, encoding="utf-8")
    status = load(MAIN / "current_status.json")
    for study in ["510300_PATTERN_DAILY_STATE_LEARNING_V3_SAVED_COMPLETION", "510300_STATE_MECHANISM_RR3_V1"]:
        if study not in status["completed_followup_studies"]:
            status["completed_followup_studies"].append(study)
    status.update({"updated_at": diagnostic["recorded_at"], "latest_goal_turn_classification": "PROGRESS_V3_COMPLETION_AND_USER_STATE_RR3_STUDY",
                   "same_condition_consecutive_no_progress_goal_turns": 0,
                   "user_requires_state_conditioned_strategies": True, "minimum_planned_net_reward_risk_ratio": 3.0,
                   "user_requires_preentry_fact_explanation": True,
                   "pending_V3_changed_or_run": True, "pending_V3_original_files_modified": False,
                   "V3_completion_directory": V3.relative_to(WORKSPACE).as_posix(), "V3_new_fits": 132, "V3_new_accounts": 6,
                   "latest_state_rr3_study": ROOT.relative_to(WORKSPACE).as_posix(), "state_rr3_new_accounts": 16,
                   "state_rr3_primary_completed_cycles": 0, "state_rr3_primary_sharpe": None,
                   "state_rr3_information_increment": diagnostic["inference_status"],
                   "remaining_research_question": "区分事前市场背景与短期触发，检验经济事实是否支持目标兑现及不同策略的相对适用性。当前短动量路由与高比例回调触发冲突；本版保持冻结，不用放松3:1、改目标或改窗口救回。",
                   "goal_achieved": False, "current_market_view": "NO_VIEW"})
    save(MAIN / "current_status.json", status)
    main_report = (MAIN / "最新研究结论.md").read_text(encoding="utf-8")
    old = "此前另有一个已冻结、未运行的普通日状态学习V3，本次没有替它运行或修改文件。"
    new = "此前已冻结但未运行的普通日状态学习V3已在独立目录按原协议补完，原文件保持不变；压力夏普0.617、年化3.90%，仍未达标。"
    assert old in main_report
    main_report = main_report.replace(old, new, 1)
    update = """用户最新进一步明确：不同市场状态、流动性和宏观对应不同策略，只做具有事前事实解释的高盈亏比机会，并确认计划净收益空间／计划损失至少3:1，使用压力成本。本轮已将这些条件写入当前约束并完成一项有限比较，未把它们仅作为口头建议。

新增状态与3:1实验完成16条账户。94个候选中9个在确认时满足3:1，7个是趋势回调、2个是急跌修复。7个回调在长期均线上方，但短期动量均为震荡，本版价格路由把它们排除；两个修复在开盘时超出买价上限。完整联合规则及同覆盖价格对照均无持仓，夏普未定义，宏观资金增量不可识别，不能把机械[0,0]区间当成无效的证明。

不加外层状态路由的同3:1对照实际成交4笔，压力净亏6784.93元、夏普-0.455，实际平均盈利／平均亏损只有0.109，目标退出为零。计划比例不是目标实现概率，也不是实际盈亏比。当前需要区分市场背景、短期触发和逻辑失效，并取得价格目标能够兑现的证据；本版结果保持冻结，不降低3:1或就地改窗口救回。

本续轮也补完原冻结V3：132次回归、66个更新时点、6条账户；量价版本压力年化3.90%、夏普0.617、最大回撤5.64%，预测误差比背景版本增加1.92%。11笔净利润46789.53元中最大一笔贡献98.34%，其余周期合计775.14元，仍未形成稳定优势。

最新详细结论见[状态、经济机制与事前3:1](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_state_mechanism_rr3_v1/研究结论.md>)和[V3原冻结实验补完](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_pattern_daily_state_learning_v3_saved_completion/研究结论.md>)。以下保留前轮全部综合结果及限制。

"""
    (MAIN / "最新研究结论.md").write_text(update + main_report, encoding="utf-8")
    print("已记录状态路由与3:1的具体失败原因，保留全部冻结结果；综合结论和最新要求已更新。")


if __name__ == "__main__":
    main()
