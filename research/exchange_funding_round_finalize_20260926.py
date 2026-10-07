"""汇总两种交易所资金价格证据及24个完整账户，保留未达成目标状态。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.exchange_repo_fixing_source_v1 as source
import research.exchange_repo_quote_crosscheck_v1 as quote
import research.exchange_bank_funding_gap_daily_v1 as fixing
import research.exchange_closing_gap_daily_v1 as closing
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    if (closing.OUT / "completed_round.json").exists():
        raise RuntimeError("本轮已有完成记录，不重复改写。")
    f, c, s, q = read(fixing.OUT / "result.json"), read(closing.OUT / "result.json"), read(source.OUT / "result.json"), read(quote.OUT / "crosscheck_result.json")
    accounts = [*f["all_accounts"], *c["all_accounts"]]
    assert len(accounts) == f["new_full_accounts"] + c["new_full_accounts"] == 24
    passed = sum(bool(row["historical_point_targets_met"]) for row in accounts)
    recomputation = []
    for module, result in [(fixing, f), (closing, c)]:
        path = module.OUT / "accounts/STRESS" / result["primary"]["policy"] / "ledger.parquet"
        verified = metrics(pd.read_parquet(path))
        for key in ["sharpe", "annual_return", "max_drawdown", "end_equity"]:
            np.testing.assert_allclose(verified[key], result["primary"][key], atol=1e-12, rtol=0)
        recomputation.append({"study": module.STUDY, "ledger": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    f_control = next(r for r in f["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == fixing.CONTROL + "__FIXED5_NO_ADD")
    c_control = next(r for r in c["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == closing.CONTROL + "__FIXED5_NO_ADD")
    f_comparison = next(r for r in f["comparisons"] if r["cost"] == "STRESS" and r["candidate"] == fixing.SIGNAL)
    c_comparison = next(r for r in c["comparisons"] if r["cost"] == "STRESS" and r["candidate"] == closing.SIGNAL)
    f_all = next(r for r in f["forecast_evaluation"] if r["subset"] == "ALL")
    c_all = next(r for r in c["forecast_evaluation"] if r["subset"] == "ALL")
    report = f"""本轮取得交易所资金价格新证据，并完成24个新的510300完整模拟账户：12个使用GC007全天定盘与银行FDR007的利差，另12个使用独立取得的真实收盘报价与FDR007的利差。共同目标仍为20万元完整账户在压力成本后夏普不低于1.2、复合年化不低于10%、最大回撤不超过10%。本轮共同目标通过数为{passed}/24；高夏普总目标保持active，尚未得到可确认稳定的策略。

先核清了两种不同的价格。

上交所当前接口用证券代码明确给出GC007定盘。本轮取得2017-05-22至2026-09-23共{s['rows']:,}个经济日期，区间内没有缺失A股交易日。与另一次官方查询重叠的8个日期数值一致。原响应中2017-12-29重复两条相同记录，保留原文后在经济日期序列中只计一次。旧曲线接口的字段名称与带代码接口对不上，已排除旧曲线，不靠猜测交换字段。

[2025年官方统计指引](https://www.sse.com.cn/lawandrules/regulations/csrcannoun/c/10787043/files/6e68fb5e2cf54a3fb3461692ae169297.pdf)的BD-II-4明确：2017-05-22及以后，定盘使用全天逐笔成交量加权平均。[现行债券交易规则第118条](https://www.sse.com.cn/lawandrules/sselawsrules2025/bond/trading/currency/c/c_20250604_10780834.shtml)则将收盘价定义为15:30之前最后一笔成交往前一小时的成交量加权价。两者不是相同观测；该现行来源在第二组模型冻结后补核，未改变数据或规则。本轮也没有把上午FDR007称作全天DR007，或把两者利差当作纯非银成本、中性利率差或流动性保险费。

定盘的历史逐日首版公布时间没有完整认证。2011旧通知记载次日发布，但该通知已列失效，不能单凭它证明当代全部日期的时钟。因此第一组固定使用“经济日后第二个交易日才可使用”的重建假设，另保留第五个交易日才使用的版本，并预先排除年末可能顺延的经济日期。这些延迟只是条件性研究假设，无法替代历史送达证据。

第一组的{f['unique_prediction_days']:,}个决策日使用每天最近两年内{f['training_rows_min']}至{f['training_rows_max']}个成熟原点，四种分布共用同一训练池。主固定五日期限压力账户从2020-01-02连续计算至2026-09-24，期末{f['primary']['end_equity']:,.2f}元，夏普{f['primary']['sharpe']:.6f}，复合年化{f['primary']['annual_return']:.4%}，最大回撤{abs(f['primary']['max_drawdown']):.4%}。同池银行资金对照夏普{f_control['sharpe']:.6f}，更迟版本夏普{f['extra_delay_diagnostic']['sharpe']:.6f}。

主定盘账户毛损益{f['primary_cash_attribution']['gross_pnl']:,.2f}元，交易费用及末端储备{f['primary_cash_attribution']['cost_and_reserve']:,.2f}元，净损益{f['primary_cash_attribution']['net_profit']:,.2f}元。相比银行资金对照的年化算术收益差为{f_comparison['annual_arithmetic_difference']:.4%}，未调整反复选择的95%区间为[{f_comparison['lower_95']:.4%}, {f_comparison['upper_95']:.4%}]。2020—2023年的差为{f_comparison['fixed_periods'][0]['annual_arithmetic_difference']:.4%}，2024年至末端为{f_comparison['fixed_periods'][1]['annual_arithmetic_difference']:.4%}，不能只保留早期。均值预测MSE相对银行资金对照改善{f_all['primary_mse_improvement'][fixing.CONTROL]:.4%}，但这没有形成合格的成本后账户。

随后取得真实收盘报价，继续执行了第二组完整账户。

新浪公开历史包含2,391行；与东方财富原始报价在2017-05-22至2026-09-24重叠{q['source_rows']:,}日，其中{q['agreed_closing_dates']:,}日收盘完全一致，{q['conflict_or_missing_closing_dates']}日不同。不同日期保持未知，没有选取更有利的提供方，也没有前填。开高低亦有不同，第二组仅使用一致收盘。两个提供方的事后对账仍不能认证历史首版，结果继续属于开发证据。

第二组使用同日收盘报价减FDR007，主版本在下一交易日使用，另延迟一天；不把前一组全天定盘提前、不把两个序列拼接。因报价已形成于当日收盘，未沿用针对定盘顺延的年末排除规则。两种延迟和银行资金对照仍以完全相同的可用日期和训练池比较。已有历史在定盘试验中已被查看，这不是新留出样本。

第二组有{c['unique_prediction_days']:,}个预测日，每日{c['training_rows_min']}至{c['training_rows_max']}个成熟训练原点。主固定五日期限压力账户期末{c['primary']['end_equity']:,.2f}元，夏普{c['primary']['sharpe']:.6f}，复合年化{c['primary']['annual_return']:.4%}，最大回撤{abs(c['primary']['max_drawdown']):.4%}。同池银行资金对照夏普{c_control['sharpe']:.6f}，额外延迟版本夏普{c['extra_delay_diagnostic']['sharpe']:.6f}。

主报价账户毛损益{c['primary_cash_attribution']['gross_pnl']:,.2f}元，费用及储备{c['primary_cash_attribution']['cost_and_reserve']:,.2f}元，净损益{c['primary_cash_attribution']['net_profit']:,.2f}元。对同池银行资金对照的年化算术差为{c_comparison['annual_arithmetic_difference']:.4%}，95%区间[{c_comparison['lower_95']:.4%}, {c_comparison['upper_95']:.4%}]；两个固定时期分别为{c_comparison['fixed_periods'][0]['annual_arithmetic_difference']:.4%}和{c_comparison['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。均值预测MSE相对对照的改善为{c_all['primary_mse_improvement'][closing.CONTROL]:.4%}。全部分布、账户、年度和滚动两年结果均保留，不能用个别表现较好的期限替换预先指定主方案。

执行与误判控制仍按原合同运行：510300.SH和人民币现金、T+1、100份、真实开盘后只能缩小买入、最多50%目标仓位、五日ES95预算2.5%、标的跌10%的账户损失预算和回撤余量、交易费用及股息分账。固定五日版本持仓期间不加仓，风险可提前减仓，10%账户回撤后本轮不恢复。两组均每日重建最近两年训练池，没有扩大训练历史或用未成熟收益选择当日模型。

原始来源未来扰动、保存分布重算和未来/过期标签扰动通过；24个账户的现金、份额、费用与T+1约束检查完成，主账户由保存账本独立重算的指标一致。风险监控仅使用固定相位的已成熟预测；是否出现尾部警报与是否具有正期望分别记录，不能把“未触发警报”当作规律有效。

目前没有独立前向观察，也没有取得当前下单资格，current_market_view仍为NO_VIEW。两种交易所资金价格的新用途均按完整结果留档。下一项先核对企业及居民中长期贷款、票据融资结构是否有可用当期原文，以及与旧社融总量用途的区别；本轮未把这一线索计算成策略结果。没有制作审核包或用户表格。
"""
    report_path = closing.OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    status = "COMPLETED_24_ACCOUNTS_NO_QUALIFIED_STRATEGY" if passed == 0 else "COMPLETED_24_ACCOUNTS_VALIDATION_STILL_REQUIRED"
    completed = {"at": now(), "status": status,
        "classification": "PROGRESS_24_NEW_ACCOUNTS_AND_TWO_EXCHANGE_FUNDING_PRICE_SERIES",
        "new_full_accounts": 24, "joint_target_pass_accounts": passed,
        "new_official_fixing_dates": s["rows"], "new_two_provider_agreed_closing_dates": q["agreed_closing_dates"],
        "conflicting_close_dates_preserved_unknown": q["conflict_or_missing_closing_dates"],
        "new_conditional_distributions": f["new_conditional_distributions"] + c["new_conditional_distributions"],
        "primary_fixing": f["primary"], "primary_closing": c["primary"], "saved_primary_metrics_recomputed": recomputation,
        "result_hashes": {str(module.OUT.relative_to(ROOT) / "result.json"): digest(module.OUT / "result.json") for module in [fixing, closing]},
        "report_sha256": digest(report_path), "goal_status": "active", "goal_achieved": False,
        "new_independent_forward_observations": 0, "current_market_view": "NO_VIEW",
        "orders_authorized": False, "review_package_created": False}
    save(closing.OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=completed["classification"], same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=closing.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_exchange_funding_price_round=completed,
        remaining_research_question="交易所定盘及一致收盘资金差的24个新完整账户已完成。下一项核对企业及居民中长期贷款/票据融资结构的首版原文与旧社融用途差异，有可用新证据后再固定检验，不重新选择本轮有利时段。",
        goal_metadata_note="本轮新增两种交易所资金价格、24个完整账户及延迟比较，是实质研究进展；高夏普目标未完成。")
    for study in ["510300_EXCHANGE_REPO_SOURCE_PROBE_V1", source.STUDY, fixing.STUDY,
                  "510300_EXCHANGE_REPO_CLOSING_QUOTE_PROBE_V1", q["study_id"], closing.STUDY]:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=closing.STUDY, latest_integrated_experiment=closing.STUDY,
        current_protocol=(closing.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(closing.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(), latest_continuation_classification=completed["classification"],
        research_execution_state=status, last_research_result=f"两种交易所资金价格24个新账户；共同目标通过{passed}个。独立前向仍为0，高夏普目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    main_status = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- EXCHANGE_FUNDING_PRICE_ROUND_20260926 -->"
    previous = main_status.read_text(encoding="utf-8")
    if marker not in previous:
        main_status.write_text(f"{marker}\n\n> 2026-09-26 交易所定盘与真实收盘资金差24个新完整账户已完成，共同目标通过{passed}个；收盘主压力夏普{c['primary']['sharpe']:.6f}、年化{c['primary']['annual_return']:.4%}。目标ACTIVE且未完成。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == report_path.read_bytes()
    print(f"两种资金价格新证据和24个账户已完成，共同目标通过{passed}个；主线保持active。", flush=True)


if __name__ == "__main__":
    run()
