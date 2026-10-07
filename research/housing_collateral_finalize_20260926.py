"""汇总住房证据与实际账户结果，保留未达标和换基敏感性。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.housing_collateral_daily_v1 as study
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    if (study.OUT / "completed_round.json").exists():
        raise RuntimeError("住房研究已汇总，不覆盖完成记录。")
    result = read(study.OUT / "result.json")
    source = read(study.source.OUT / "result.json")
    p = result["primary"]
    control_name = study.policy_name(study.CONTROL, study.SCOPES[0], 20)
    short_name = study.policy_name(study.SIGNAL, study.SCOPES[0], 5)
    strict_name = study.policy_name(study.SIGNAL, study.SCOPES[1], 20)
    control = next(m for m in result["all_accounts"] if m["policy"] == control_name and m["cost"] == "STRESS")
    short = next(m for m in result["all_accounts"] if m["policy"] == short_name and m["cost"] == "STRESS")
    strict = next(m for m in result["all_accounts"] if m["policy"] == strict_name and m["cost"] == "STRESS")
    increment = next(r for r in result["comparisons"] if r["purpose"] == "HOUSING_INFORMATION_INCREMENT" and r["scope"] == study.SCOPES[0] and r["horizon"] == 20 and r["cost"] == "STRESS")
    strict_increment = next(r for r in result["comparisons"] if r["purpose"] == "HOUSING_INFORMATION_INCREMENT" and r["scope"] == study.SCOPES[1] and r["horizon"] == 20 and r["cost"] == "STRESS")
    horizon = next(r for r in result["comparisons"] if r["purpose"] == "TWENTY_MINUS_FIVE_HORIZON" and r["cost"] == "STRESS")
    schedules = {r["scope"]: r for r in result["schedules"]}
    main_schedule, strict_schedule = schedules[study.SCOPES[0]], schedules[study.SCOPES[1]]
    seen, duplicate_paths = {}, []
    for item in result["all_accounts"]:
        ledger = pd.read_parquet(study.OUT / "accounts" / item["cost"] / item["policy"] / "ledger.parquet")
        fresh = metrics(ledger)
        for key in ["end_equity", "profit", "annual_return", "sharpe", "max_drawdown", "mean_exposure"]:
            if fresh[key] is None or item[key] is None:
                assert fresh[key] is None and item[key] is None
            else:
                np.testing.assert_allclose(fresh[key], item[key], atol=1e-10, rtol=0)
        signature = hashlib.sha256(ledger[["equity", "shares", "cash", "net_return"]].to_numpy(float).tobytes()).hexdigest()
        key = item["policy"] + "/" + item["cost"]
        if signature in seen:
            duplicate_paths.append({"path": key, "same_as": seen[signature]})
        else:
            seen[signature] = key
    assert result["new_full_accounts"] == 12 and len(result["all_accounts"]) == 12
    attribution = result["primary_cash_attribution"]
    monitor = [r for r in result["mature_monitors"] if r["scope"] == study.SCOPES[0] and r["model"] == study.SIGNAL]
    latest = source["latest_source"]
    report = f"""住房资产与抵押品这条新证据已经完成原始来源重建和12个完整账户。主方案用已公布的二手住宅环比下跌城市占比，改善价格与银行资金条件下的20日收益判断，并每天按五日尾部风险约束仓位。主压力账户净夏普{study.sharpe_text(p['sharpe'])}、年化收益{p['annual_return']:.4%}、最大回撤{abs(p['max_drawdown']):.4%}；12个账户共同目标通过{result['joint_target_pass_accounts']}个。高夏普目标尚未完成，继续保持active；独立前向结果为0。

本轮取得2021年1月至2026年8月68份国家统计局原月报，逐月保存70城二手住宅环比原值，共4,760个城市月观察。目录已定位的60份与原页公布日核对，2021年前8个月依据原页时钟；统一公布日日末才可用。桌面和移动重复表格逐值一致，未混入新房、同比或面积分类表。最早的2018年探查页只用于辨认口径，没有混入模型训练。原始历史网页不是已认证的不可变首版。

这项指标只表示公布的一位小数环比中，下跌城市占70城的比例。它不是全国住房总市值损失率、银行抵押贷款损失率或股票资金流。调查范围为市辖区，二手住宅采用重点与典型调查结合；无成交可按价格无变化处理，公布值持平也可能受四舍五入影响。故“持平城市多”不能直接解释为流动性充足。抵押品和财富压力可能降低风险承受力，也可能伴随政策缓解或资产重新配置，方向由当时历史条件分布估计，没有事后指定“房价跌就买/卖股票”。

[2021年原报告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900997.html)与[2026年1月原报告](https://www.stats.gov.cn/sj/zxfb/202602/t20260213_1962617.html)说明，换基调整了住宅类别权重。官方说明的平均同比影响不能直接证明“环比下跌城市数”不受影响。主方案使用各期实际公布的同名环比广度，并保留跨基期不完全可比的限制；另做严格只使用同基期历史的20日模型和完全同池价格资金对照，不在看到结果后选一个版本。

这次同时检验宏观作用的时间尺度。主方案事前定为20个交易日，5日作为期限对照；不是看到5日结果不好才修改期限。两个期限始终使用同一批已经完成20日结果的训练原点，退出开盘必须严格早于当前09:00决策。每天只用最近两日历年、至少252个成熟原点，来源公布时间也在两年内；固定126近邻，训练内标准化clip5。价格压力、趋势、波动结构和银行资金日历残差为共同基础，只新增一个住房广度字段。月报源龄上限45自然日。没有搜索阈值、方向、近邻数或最佳持有天数。

主样本共有{result['unique_prediction_days']:,}个预测日。每个主训练池{main_schedule['training_rows_min']}—{main_schedule['training_rows_max']}个日原点，但实际只来自{main_schedule['release_months_min']}—{main_schedule['release_months_max']}个不同月报；日原点重叠和月报重复不能被当成数百次独立宏观事件。主方案首次预测为{main_schedule['first_prediction'][:10]}，完整2020-01-02至2026-09-24账本仍保留此前无信息/训练不足时的现金日。这些现金日不是已经完成市场验证的稳定期。

20万元主压力账户期末{p['end_equity']:,.2f}元，净损益{p['profit']:,.2f}元。毛损益{attribution['gross_pnl']:,.2f}元，费用与退出储备{attribution['cost_and_reserve']:,.2f}元，平均敞口{p['mean_exposure']:.4%}。同池同20日期限的价格资金对照净夏普{study.sharpe_text(control['sharpe'])}、年化{control['annual_return']:.4%}、回撤{abs(control['max_drawdown']):.4%}。住房模型5日对照净夏普{study.sharpe_text(short['sharpe'])}、年化{short['annual_return']:.4%}、回撤{abs(short['max_drawdown']):.4%}。

住房信息在20日期限上的年化算术收益增量为{increment['daily_blocks']['annual_arithmetic_difference']:.4%}，20日区块95%区间为[{increment['daily_blocks']['lower_95']:.4%}, {increment['daily_blocks']['upper_95']:.4%}]，完整月报公布间隔区间为[{increment['release_blocks']['lower_95']:.4%}, {increment['release_blocks']['upper_95']:.4%}]。预定2020—2023和2024—末端两个时期增量分别为{increment['fixed_periods'][0]['annual_arithmetic_difference']:.4%}、{increment['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。2026新基期首次可用于交易日决策以来的增量为{increment['fixed_periods'][2]['annual_arithmetic_difference']:.4%}。两种区块都未校正全部历史选择，月报区块也不完整保留相邻月份的持续性。

同一住房信息的20日减5日期限收益差为{horizon['daily_blocks']['annual_arithmetic_difference']:.4%}，95%区间[{horizon['daily_blocks']['lower_95']:.4%}, {horizon['daily_blocks']['upper_95']:.4%}]。这只是期限差异，不能冒充住房信息的新增价值。

严格同基期的住房20日压力账户净夏普{study.sharpe_text(strict['sharpe'])}、年化{strict['annual_return']:.4%}、回撤{abs(strict['max_drawdown']):.4%}。它相对自身同池对照的年化算术收益增量为{strict_increment['daily_blocks']['annual_arithmetic_difference']:.4%}，20日区块95%区间[{strict_increment['daily_blocks']['lower_95']:.4%}, {strict_increment['daily_blocks']['upper_95']:.4%}]，月报间隔区间[{strict_increment['release_blocks']['lower_95']:.4%}, {strict_increment['release_blocks']['upper_95']:.4%}]。末端同基期状态为{strict_schedule['last_status']['status']}，成熟训练原点{strict_schedule['last_status']['n_train']}个；不足时没有放宽252门槛。主开发增量条件通过状态为{result['development_increment_gate']}，主账户滚动两年共同目标通过{result['primary_rolling_two_year_joint_passes']}次。

所有账户按实际开盘成交和压力费用运行，保持T+1、100份、现金股息分账、最多50%目标仓位和10%回撤停止规则。20日均值与方差用于20日期限的收益评价；风险约束始终使用五日ES95，不把20日ES当作五日ES。持仓不加仓，风险可提前减仓；到期请求卖出仍受成交限制。累计连续跳空和预测错误仍可能使事前预算失准。12个新计算账户实际有{len(seen)}条不同净值路径，重复路径{len(duplicate_paths)}条，详细对应已保存。

五日与二十日主监控分别有{monitor[0]['nonoverlap_rows']}、{monitor[1]['nonoverlap_rows']}个成熟非重叠原点，可评估时点分别{monitor[0]['assessable_rows']}、{monitor[1]['assessable_rows']}个，警报分别{monitor[0]['alerts']}、{monitor[1]['alerts']}个。无警报不能证明收益优势，二十日非重叠样本更少。保存预测重算、未来/过期标签扰动、两期限分红边界、5日期限复现和完整账户恒等式均已完成。

最新原报告为{latest['stat_month']}，原公布时间{latest['published_at']}，70城中公布二手住宅环比下跌{latest['second_hand_down_count']}城。这个已公布宏观事实不等于当前可交易策略，current_market_view仍为NO_VIEW。本轮来源、训练和账户计算均已结束，没有执行真实订单，没有制作审核包或用户表格。研究总目标继续，所有结果与未达标状态保留。
"""
    path = study.OUT / "本轮研究结论.md"
    path.write_text(report, encoding="utf-8")
    status = "COMPLETED_HOUSING_COLLATERAL_TWELVE_ACCOUNTS_NO_QUALIFIED_STRATEGY" if not result["joint_target_pass_accounts"] else "COMPLETED_HOUSING_COLLATERAL_ACCOUNTS_VALIDATION_PENDING"
    classification = "PROGRESS_HOUSING_BREADTH_TWO_HORIZONS_AND_REBASE_SENSITIVITY"
    receipt = {"at": now(), "study_id": study.STUDY, "status": status, "classification": classification,
        "source_months": 68, "city_observations": 4760, "new_accounts": 12,
        "distinct_paths": len(seen), "duplicate_paths": duplicate_paths, "primary": p,
        "information_increment": increment, "strict_base_increment": strict_increment,
        "joint_target_pass_accounts": result["joint_target_pass_accounts"],
        "result_sha256": digest(study.OUT / "result.json"), "report_sha256": digest(path),
        "goal_status": "active", "goal_achieved": False, "new_independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(study.OUT / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=study.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_housing_collateral_round=receipt,
        remaining_research_question="住房下跌范围与20日/5日期限已形成12个完整账户，严格同基期敏感性已完成；继续根据信息增量与收益分布寻找机制，不用更换口径或挑期限营救本轮结果。",
        goal_metadata_note="本轮取得68个月住房原表，新增12账户并完成换基敏感性，属于实质研究进展；独立前向仍不足。")
    state["next_evidence_lead"] = {"study": study.source.STUDY, "status": "SOURCES_AND_TWELVE_ACCOUNT_EXPERIMENT_COMPLETED", "completed_receipt": (study.OUT / "completed_round.json").relative_to(ROOT).as_posix()}
    for name in [study.source.probe.STUDY, study.source.STUDY, study.STUDY]:
        if name not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(name)
    save(MAIN / "current_status.json", state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=study.STUDY, latest_integrated_experiment=study.STUDY,
        current_protocol=(study.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(study.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=path.relative_to(ROOT).as_posix(), latest_continuation_classification=classification,
        research_execution_state=status,
        last_research_result=f"住房广度完成68个月原表、12新账户/{len(seen)}条不同路径；主20日压力夏普{study.sharpe_text(p['sharpe'])}、年化{p['annual_return']:.4%}、回撤{abs(p['max_drawdown']):.4%}；总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- HOUSING_COLLATERAL_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 住房广度68个月原表、12新账户/{len(seen)}条不同路径完成；共同目标通过{result['joint_target_pass_accounts']}个，独立前向0，总目标ACTIVE。见[本轮结论]({path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == path.read_bytes()
    print(f"住房研究结论已登记：12个账户/{len(seen)}条不同路径，总目标仍active。", flush=True)


if __name__ == "__main__":
    run()
