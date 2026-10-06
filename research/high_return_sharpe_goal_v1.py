"""登记用户的绝对收益与夏普目标，核算保存账户差距；不改策略或回测。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_high_return_sharpe_goal_v1"
CONFIG = ROOT / "config/510300_high_return_sharpe_goal_v1.json"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
FINANCIAL = ROOT / "reports/research/510300_all_factor_macro_earnings_account_v1"
METRICS = FINANCIAL / "results/全部20账户_四场景同口径比较.parquet"
DOCS = ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")
NAMES = {"A_SAVED_WEIGHT": "原A", "TECH_COMMON": "同池价量", "TECH_MACRO_COMMON": "价量＋宏观",
         "TECH_EARNINGS_COMMON": "价量＋原始盈利", "TECH_MACRO_EARNINGS_JOINT": "主三源联合"}
PERIODS = ("2015_2019", "2020_2026")
COSTS = ("BASE", "STRESS")
TARGETS = {"net_cagr_minimum": .10, "net_sharpe_minimum": 1.5, "max_drawdown_maximum": .10,
           "p_times_b_strict_minimum": 1., "standard_net_expectancy_strict_minimum": 0.,
           "mean_cycle_net_return_strict_minimum": 0.}
FINANCIAL_FIELDS = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
                    "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics")
FORWARD_FIELDS = ("forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
                  "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
                  "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
                  "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation", "next_experiment")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def evaluate(metric):
    """经济点值通过与独立验证分开；未定义的夏普/实际盈亏比不能通过。"""
    rules = (
        ("net_cagr", ">=", TARGETS["net_cagr_minimum"], "净年化不足10%"),
        ("net_sharpe", ">=", TARGETS["net_sharpe_minimum"], "净夏普不足1.5"),
        ("max_drawdown", "<=", TARGETS["max_drawdown_maximum"], "实际回撤超过10%"),
        ("p_times_b", ">", TARGETS["p_times_b_strict_minimum"], "实际净胜率乘盈亏比未严格大于1"),
        ("standard_expectancy_loss_units", ">", 0., "标准净期望未严格为正"),
        ("mean_cycle_net_return", ">", 0., "实际完成周期净均值未严格为正"),
    )
    reasons, checks = [], {}
    for field, comparison, threshold, reason in rules:
        value = metric.get(field)
        if not finite(value):
            passed = False
            reasons.append(field + "未定义或非有限")
        else:
            value = float(value)
            passed = value >= threshold if comparison == ">=" else value <= threshold if comparison == "<=" else value > threshold
            if field == "max_drawdown" and value < 0:
                passed = False
                reasons.append("实际回撤符号不符合损失比例定义")
            elif not passed:
                reasons.append(reason)
        checks[field + "_pass"] = bool(passed)
    return {**checks, "historical_economic_pass": all(checks.values()), "failed_reasons": reasons,
            "independent_validation": "NOT_ESTABLISHED", "validated_goal_pass": False}


def freeze():
    if CONFIG.exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("本绝对目标已经登记，不覆盖原合同。")
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"study_id": "510300_HIGH_RETURN_SHARPE_GOAL_V1", "registered_at": now(), "registration": "TECH.R261",
              "user_instruction": "我必须要实现高夏普和高收益率",
              "user_confirmed_target": "净年化≥10%、净夏普≥1.5、最大回撤≤10%（建议）",
              "authority": "当前thread用户实际回复，非从旧委托推定", "targets": TARGETS,
              "capital_cny": 200000, "assets": ["510300.SH", "CASH_CNY"], "long_first": True,
              "bars": ["DAILY", "PREVIOUS_COMPLETED_WEEK"], "annual_days": 252,
              "cash_return_and_sharpe_reference_assumption": 0., "calendar": "全部账户交易日日收益，包含空仓、费用和自然期末持仓",
              "costs_each_side": {"BASE": {"commission": .0002, "slippage": .0005},
                                   "STRESS": {"commission": .0004, "slippage": .001}},
              "minimum_fee_cny": 5, "lot": 100, "tick": .001, "execution": "保持原次开盘、T+1及分红应收/到账规则",
              "risk_contract": "保持原最高50%股票请求与ES/跳空预算、10%收盘回撤停止新入场；实测回撤仍须<=10%",
              "frequency": "软目标，不设逐年交易配额", "current_strategy_parameters_changed": False,
              "historical_scenario_gate": "原两时期×两费用四场景均符合所有经济要求，不能取其中最好时期；这是开发检验，不等于独立验证",
              "relative_improvement_retained": "新模型仍须在同资金/费用/风险/日历下同时改善原A的净CAGR与净Sharpe",
              "independent_requirement": "真实事前登记、新样本、规则不变的完整账户独立验证；历史切分/滚动/区块区间不升级为独立结果",
              "old_forward_protocol": "原1008真实交易日、至少30完整周期及各至少5盈亏的机制比较合同保持，不追溯修改；它本身不是全策略独立成功",
              "legacy_target_reconciliation": "本目标从用户确认起生效；旧1.2/1.5、242/252日和各研究的原裁决均保存，不能混口径择优",
              "goal_achieved": False, "orders_authorized": False}
    write(CONFIG, config, exclusive=True)
    sources = [METRICS, FINANCIAL / "summary.json", FINANCIAL / "protocol.json", FINANCIAL / "verification_receipt.json",
               CONTEXT / "state.json", CONTEXT / "active_goal_effective_requirements.json", CONFIG,
               Path(__file__).absolute(), ROOT / "tests/test_high_return_sharpe_goal_v1.py"]
    write(OUT / "protocol.json", {"at": now(), "registration": "TECH.R261", "decision": "TECH.R262",
          "purpose": "接受新明确目标，完整核算当前20保存账户缺口；不重新拟合或模拟",
          "known_before_registration": "R256及原A指标已见，本轮是事后目标核对，不是新预测实验",
          "sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
          "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_strategy_configurations": 0,
          "new_market_requests": 0, "old_failures_retained": True}, exclusive=True)
    print("已登记用户明确的10%净年化、1.5净夏普和10%最大回撤目标；原策略未修改。", flush=True)


def run():
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError("目标核对登记后输入改变：" + source["path"])
    write(OUT / "run_started.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    saved = pd.read_parquet(METRICS)
    keys = ["period", "cost", "policy"]
    actual = set(map(tuple, saved[keys].to_numpy()))
    expected = {(p, c, m) for p in PERIODS for c in COSTS for m in NAMES}
    if len(saved) != 20 or actual != expected or saved.duplicated(keys).any():
        raise ValueError("原两时期、两费用及五政策的完整20账户结构不同。")
    rows = []
    for item in saved.to_dict("records"):
        gate = evaluate(item)
        required_wealth = 200000 * (1 + TARGETS["net_cagr_minimum"]) ** (int(item["days"]) / 252)
        rows.append({**item, "目标净年化差_百分点": 100 * (TARGETS["net_cagr_minimum"] - item["net_cagr"]),
                     "目标净夏普差": TARGETS["net_sharpe_minimum"] - item["net_sharpe"] if finite(item["net_sharpe"]) else None,
                     "目标期末净财富": required_wealth, "目标期末财富缺口_元": required_wealth - item["ending_equity"],
                     **{k: v for k, v in gate.items() if k != "failed_reasons"}, "失败原因": "；".join(gate["failed_reasons"])})
    result = pd.DataFrame(rows)
    result.to_parquet(OUT / "全部20保存账户_新绝对目标差距.parquet", index=False)
    result.to_csv(OUT / "全部20保存账户_新绝对目标差距.csv", index=False, encoding="utf-8-sig")
    policies = [{"policy": name, "four_scenes_economic_pass": bool(block.historical_economic_pass.all()),
                 "passed_scenes": int(block.historical_economic_pass.sum()), "independent_validation": "NOT_ESTABLISHED"}
                for name, block in result.groupby("policy", sort=False)]
    def clean(records):
        return json.loads(pd.DataFrame(records).to_json(orient="records", force_ascii=False, double_precision=15))
    summary = {"at": now(), "decision": "TECH.R262", "status": "COMPLETED_NEW_ABSOLUTE_TARGET_GAP_NOT_ACHIEVED",
               "confirmed_targets": TARGETS, "scope": "最新R256的20保存账户，非全repo穷尽筛选",
               "historical_economic_pass_accounts": int(result.historical_economic_pass.sum()), "accounts": len(result),
               "four_scene_policies_passed": sum(r["four_scenes_economic_pass"] for r in policies), "all_five_policy_results": policies,
               "sharpe_undefined_accounts": int(result.net_sharpe.isna().sum()),
               "A_pressure_both_periods": clean(result[result.policy.eq("A_SAVED_WEIGHT") & result.cost.eq("STRESS")]),
               "recent_pressure_all_policies": clean(result[result.period.eq("2020_2026") & result.cost.eq("STRESS")]),
               "latest_actual_financial_decision": "TECH.R256", "new_accounts": 0, "new_fits": 0, "new_labels": 0,
               "new_market_requests": 0, "new_strategy_configurations": 0,
               "all_history_through": "2026-09-30", "history_role": "DEVELOPMENT_CALIBRATION",
               "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False,
               "feasibility_claim": "当前保存账户未通过，不是单标的目标不可能的穷尽证明；不能承诺未来表现",
               "goal_achieved": False, "orders_authorized": False}
    write(OUT / "summary.json", summary, exclusive=True)
    lines = ["# 510300高收益与高夏普：明确目标及实际差距", "",
             "用户已明确：20万元完整账户，扣费后净年化≥10%、净夏普≥1.5、最大回撤≤10%。包含全部空仓日，252日年化、现金/无风险比较率按原合同假设0。保留实际净pB>1、标准净期望和周期净均值为正、次数软目标及独立验证。", "",
             "本次核对最新R256的全部20保存账户，不拟合、不模拟新策略。新目标核对不是重新裁决原冻结协议，也不是穷尽所有可能策略。", "",
             "|时期／费用／账户|净年化|净夏普|实际回撤|实际净pB|完成周期|年化差距百分点|夏普差距|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in result.iterrows():
        s = f"{row.net_sharpe:.4f}" if finite(row.net_sharpe) else "未定义"
        b = f"{row.p_times_b:.4f}" if finite(row.p_times_b) else "未定义"
        gap = f"{row['目标净夏普差']:.4f}" if finite(row["目标净夏普差"]) else "未定义"
        lines.append(f"|{row.period}/{row.cost}/{NAMES[row.policy]}|{row.net_cagr:.4%}|{s}|{row.max_drawdown:.4%}|{b}|{int(row.completed_cycles)}|{row['目标净年化差_百分点']:.4f}|{gap}|")
    lines.extend(["", f"单账户通过{summary['historical_economic_pass_accounts']}/20，四场景政策通过{summary['four_scene_policies_passed']}/5；8个零交易账户的夏普未定义，不写作高夏普。真实独立合格策略0，目标未完成。", "",
                  "## 原A与目标的实际距离", ""])
    for row in summary["A_pressure_both_periods"]:
        lines.append(f"- {row['period']}压力费用：净年化{row['net_cagr']:.4%}、净夏普{row['net_sharpe']:.4f}，分别缺{row['目标净年化差_百分点']:.4f}个百分点与{row['目标净夏普差']:.4f}。相同{row['days']}交易日达到10%净CAGR须期末财富{row['目标期末净财富']:.2f}元，原实际{row['ending_equity']:.2f}元，缺{row['目标期末财富缺口_元']:.2f}元；这是目标算术，不是可执行增量策略。")
    lines.extend(["", "## 接下来如何推进", "",
                  "先解释具体上涨的启动、延续与拥挤阶段，再登记可提前识别的新机制。已经保存的2024/9/24与9/27启动盈利、10/8后进入亏损和所有其他反例均保留；不能根据三例改旧树或增加只服务于这些日期的条件。", "",
                  "全因素12类83项目录继续用于共同讨论，未知不补零、同一事件多报道不重复加票。解释分、未经校准的预测分和实际交易胜率分别报告。新增因子是否提高信息、点位与完整账户必须分别验证。", "",
                  "现有最具体的新信息入口是已经登记的官方ETF份额真实版本：10/8首个未来版本、10/9第二个相邻版本实际到达后，才可能从10/12起观察净份额变化与上涨阶段关系。份额变化不是成交主动买盘或人民币净流入，公布/取得在收盘之后则不能用于当天15:05决定；不能把未来日期或配置文件当已取得数据。这一入口尚未有两个新版本，也未准入数值策略。", "",
                  "新完整机制需在读新结果前固定一个主候选、原A及必要去组对照、相同资金风险和两费用；所有成功、失败、未知和无交易年保存。先检验原四场景完整账户，再对真正新样本独立检验。任何一个经济或独立性条件失败都不能宣布目标实现。原1008日机制比较协议保留；未建立自动采集或交易任务。", "",
                  "现有价量、宏观、原始盈利和期权PCR等信息已分别试过多种用途，不能重新命名为新增来源。若没有完整的新机制或真实新样本，应明确证据不足；重复状态核对、不断改变参数或收益/夏普定义不能填补缺口。", "",
                  "增加固定仓位不提供新的预测优势：在零现金收益、忽略费用/约束且每日超额收益恰为常数倍的数学假设下，正比例放大使均值与标准差同比增加，夏普不变。真实账户还有整手、费用和路径约束，不能将这种比例关系当成账户回测。", "",
                  "## 裁决和边界", "",
                  "接受明确的绝对目标与完整差距测量；拒绝将当前任何一个保存账户宣布为实现高收益高夏普。最新实际金融仍R256，原失败不改。目标服务的原blocked状态另保留，不能把本次合同登记冒充系统已恢复或模型进展。", "",
                  "只有合格的新信息完整用途、确证来源/实现错误或真正新独立样本出现时，另登记有限实验。本核对一次结束，不据差距启动杠杆、扩大资产、期权损益或同家族参数优化。研究目标明确不等于未来结果保证。", "",
                  "直接来源：`config/510300_high_return_sharpe_goal_v1.json`；原R256 `summary.json`与`results/全部20账户_四场景同口径比较.parquet`；R258全部路径诊断；原官方份额观察合同与原前瞻协议。"])
    (OUT / "明确目标_保存账户差距与推进路线.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "run_completed.json", {"at": now(), "terminal": True, "status": summary["status"],
          "goal_achieved": False, "new_accounts": 0}, exclusive=True)
    print(f"全部20保存账户核对完成：绝对经济目标{summary['historical_economic_pass_accounts']}/20，四场景政策{summary['four_scene_policies_passed']}/5；目标未实现。", flush=True)


def close_once():
    summary = read(OUT / "summary.json")
    if not read(OUT / "run_completed.json")["terminal"]:
        raise RuntimeError("差距核对尚未结束。")
    write(OUT / "close_started.json", {"at": now(), "project_updates": 1}, exclusive=True)
    state_path = CONTEXT / "state.json"
    previous = read(state_path)
    preserved = {k: copy.deepcopy(previous.get(k)) for k in FINANCIAL_FIELDS + FORWARD_FIELDS + ("independent_official_share_observer",)}
    write(OUT / "state_before_TECH_R262.json", previous, exclusive=True)
    git_before = subprocess.check_output(["git", "status", "--porcelain=v1", "-uno"], cwd=ROOT)
    effective_path = CONTEXT / "active_goal_effective_requirements.json"
    effective = read(effective_path)
    write(OUT / "effective_requirements_before_TECH_R262.json", effective, exclusive=True)
    effective.update(at=now(), latest_user_instruction="我必须要实现高夏普和高收益率；净年化≥10%、净夏普≥1.5、最大回撤≤10%",
                     latest_user_account_instruction="净年化≥10%、净夏普≥1.5、最大回撤≤10%（建议）",
                     explicit_absolute_account_targets=TARGETS, absolute_target_contract=relative(CONFIG),
                     account_quality_objective="20万元完整扣费全日历账户同时符合净CAGR>=10%、净Sharpe>=1.5、实际DD<=10%；原实际净pB/期望及独立性要求保持",
                     goal_achieved=False)
    write(effective_path, effective)
    updated = copy.deepcopy(previous)
    updated.update(updated_at=now(), latest_user_instruction=effective["latest_user_instruction"],
                   current_absolute_goal_targets=TARGETS, current_absolute_goal_contract=relative(CONFIG),
                   latest_completed_study=relative(OUT), latest_result=relative(OUT / "summary.json"),
                   latest_report=relative(OUT / "明确目标_保存账户差距与推进路线.md"),
                   latest_research_status=summary["status"], latest_absolute_goal_gap=relative(OUT / "summary.json"),
                   latest_progress="登记用户确认的10%/1.5/10%绝对验收；最新20保存账户0达标，未运行新模型或策略",
                   current_unmet_evidence="当前20保存账户没有实现10%净CAGR、1.5净Sharpe及10%回撤联合门；原新来源用途与真实独立样本尚未具备",
                   current_priority="明确绝对目标下的新完整信息机制与实际新样本；旧来源用途和冻结失败保持",
                   goal_achieved=False, last_target_registration_classification="REQUIREMENTS_AND_SAVED_GAP_ONLY_NOT_MODEL_IMPROVEMENT")
    write(state_path, updated)
    A = {r["period"]: r for r in summary["A_pressure_both_periods"]}
    block = f"""### TECH.R261—R262：用户明确高收益/高夏普绝对目标与完整差距（2026-10-06）

**当前正式目标：20万元完整扣费账户，净CAGR≥10%、净Sharpe≥1.5、实际最大回撤≤10%；全部空仓日计入。目标未实现。** 用户在当前thread实际确认这组数值，原‘fresh_explicit_numeric_target=null’仅是旧事实，后续用新绝对目标合同；不追溯改旧协议。

假设→‘高’需要明确验收；最近保存账户可能具备部分优势，但须同时满足收益、夏普、回撤、实际净pB/期望及独立验证。
验证方法→R261登记用户目标，R262只核原R256全部20保存账户（五政策×两时期×两费用）、保留零交易/未定义；原252日、费用、20万元、自然末端、风险和日历不改。0拟合/新账户/标签/策略配置/网络请求。属于已见结果后的目标差距核对，不是新Alpha或独立实验。
结果→经济点值通过{summary['historical_economic_pass_accounts']}/20、四场景政策通过{summary['four_scene_policies_passed']}/5；8个零交易账户净夏普未定义。原A近期压力净年化{A['2020_2026']['net_cagr']:.4%}、净夏普{A['2020_2026']['net_sharpe']:.6f}，缺{A['2020_2026']['目标净年化差_百分点']:.4f}个百分点/{A['2020_2026']['目标净夏普差']:.6f}；较早{A['2015_2019']['net_cagr']:.4%}/{A['2015_2019']['net_sharpe']:.6f}也未通过。最新实际金融仍R256拒绝，新独立合格0；未证明所有可能策略都不可能，也不能承诺未来达标。
接受/拒绝→接受用户目标及完整差距测量；拒绝宣称任何当前保存账户实现目标。历史时期、242/252日、归一化单笔与完整资金不能择优混用。原全因素12类83项讨论、所有失败及未知继续保留；未校准预测分不当实际胜率。独立验证、净pB>1、标准净期望/净均值>0、频率软目标保持。
下一步→优先实际新增且完整的需求信息与上涨启动/延续/拥挤阶段关系；已登记官方ETF份额未来真实版本是具体入口，需两个相邻新版本到达且通过其原时钟才能观察净变化，尚未准入策略。公开催化旧65目录、旧PCR、宏观/盈利树等失败不重开。新机制先解释全部成功/失败，再于新结果前冻结唯一主候选和完整同口径账户；同时满足新绝对门和相对A两指标改善，之后仍须真正新样本独立检验。当前已准入待跑数值策略0，自动采集/交易0；原1008日机制比较和份额来源合同不追溯修改。
目标服务→原blocked实际服务状态保留。本轮用户重申目标并确认数值，但没有工具可将blocked自行改为active；不把合同或差距文档当收益/模型进展，不重启三轮状态循环。
证据→`config/510300_high_return_sharpe_goal_v1.json`；`reports/research/510300_high_return_sharpe_goal_v1/summary.json`、`全部20保存账户_新绝对目标差距.parquet`、`明确目标_保存账户差距与推进路线.md`。旧文档正文、金融五字段、前瞻十三字段及官方份额合同保持。

"""
    for name in DOCS:
        path = ROOT / "docs" / name
        content = path.read_bytes()
        (OUT / (path.stem + "_before_TECH_R262.md")).write_bytes(content)
        title, body = content.split(b"\n", 1)
        path.write_bytes(title + b"\n" + ("\n" + block).encode("utf-8") + body)
    final = read(state_path)
    if any(final.get(k) != v for k, v in preserved.items()):
        raise RuntimeError("只登记新目标时原金融/前瞻/份额事实意外变化。")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "-uno"], cwd=ROOT) != git_before:
        raise RuntimeError("用户已有Git跟踪状态变化。")
    write(OUT / "project_state_update_receipt.json", {"at": now(), "state_updates": 1,
          "user_absolute_targets_registered": True, "existing_financial_forward_share_fields_preserved": True,
          "goal_service_status_unchanged": final.get("goal_tool_status_confirmed"),
          "tracked_git_status_preserved": True, "documents": list(DOCS), "goal_achieved": False}, exclusive=True)
    print("新绝对目标和实际差距已写入长期事实，原金融/前瞻/份额合同及旧策略保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="登记510300明确的绝对收益夏普目标，并只核保存账户。")
    parser.add_argument("command", choices=("freeze", "run", "close"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "close": close_once}[args.command]()


if __name__ == "__main__":
    main()
