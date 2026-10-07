"""合并近期新增净值，保存折价补偿结论及后续研究边界。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, digest, now
import research.etf_discount_compensation_probe_v1 as probe
import research.etf_nav_recent_source_completion_v1 as source

OUT = ROOT / "reports/research/510300_etf_discount_compensation_extension_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
STUDY = "510300_ETF_DISCOUNT_COMPENSATION_EXTENSION_V1"


def run():
    prior = read(probe.OUT / "result.json")
    addition = read(source.OUT / "result.json")
    assert addition["status"] == "NEW_NAV_EXTENSION_COMPLETED"
    assert addition["provider_difference_or_missing_rows"] == 0 and addition["overlap_difference_rows"] == 0
    paths = [Path(__file__), probe.OUT / "result.json", probe.OUT / "results/observed_discount_and_attribution.parquet",
             source.OUT / "result.json", source.OUT / "matched_new_nav.parquet"]
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    save(OUT / "completion_plan.json", {"at": now(), "study_id": STUDY, "inputs": input_hashes,
        "action": "只按已冻结费用和定义纳入新取得31日净值；原测量与原源文件不修改，不搜索阈值。",
        "new_accounts": 0, "new_fits": 0, "source_first_publication_verified": False}, True)
    frame = pd.read_parquet(probe.OUT / "results/observed_discount_and_attribution.parquet")
    original = frame.copy(deep=True)
    fresh = pd.read_parquet(source.OUT / "matched_new_nav.parquet")
    fresh = fresh[fresh.new_nav_date & fresh.close.notna()].copy()
    mapped = fresh.set_index("date")
    new_mask = frame.date.isin(mapped.index)
    assert int(new_mask.sum()) == addition["new_matched_nav_dates"] and not frame.loc[new_mask, "nav_known_in_archive"].any()
    for column in ["unit_nav", "nav_eastmoney", "nav_sina"]:
        frame.loc[new_mask, column] = frame.loc[new_mask, "date"].map(mapped.unit_nav).to_numpy()
    frame.loc[new_mask, "source_dual_difference"] = 0.
    frame.loc[new_mask, "nav_known_in_archive"] = True
    frame["premium"] = frame.close / frame.unit_nav - 1
    frame["discount_recovery_gross_fraction"] = frame.unit_nav / frame.close - 1
    for index, row in frame[new_mask].iterrows():
        for name, cost in probe.market_source.distribution.COSTS.items():
            calculated = probe.space(row.close, row.unit_nav, cost)
            for key, value in calculated.items():
                frame.loc[index, name + "_" + key] = value
    cash = frame.dividend.rolling(5, min_periods=5).sum().shift(-5)
    frame["nav_component_close5"] = (frame.unit_nav.shift(-5) - frame.unit_nav + cash) / frame.close
    spread = frame.close - frame.unit_nav
    frame["spread_component_close5"] = (spread.shift(-5) - spread) / frame.close
    frame["complete_decomposition5"] = frame.nav_known_in_archive & frame.nav_known_in_archive.shift(-5, fill_value=False)
    valid = frame.complete_decomposition5
    np.testing.assert_allclose(frame.loc[valid, "etf_return_close5"],
        frame.loc[valid, "nav_component_close5"] + frame.loc[valid, "spread_component_close5"], atol=1e-12, rtol=0)
    frame.to_parquet(OUT / "results/extended_observed_discount.parquet", index=False)
    # 新日期不改变原压力空间为正的任何日期或已成熟归因，才复用原统计。
    old_eligible = original.STRESS_static_positive.eq(True)
    eligible = frame.STRESS_static_positive.eq(True)
    assert old_eligible.equals(eligible)
    columns = ["date", "etf_return_close5", "nav_component_close5", "spread_component_close5", "delayed_return_open5"]
    pd.testing.assert_frame_equal(original.loc[old_eligible, columns], frame.loc[eligible, columns])
    current = frame[frame.date.between("2020-01-02", "2026-09-24")]
    assert current.nav_known_in_archive.all()
    current_events = current[current.STRESS_static_positive.eq(True)]
    recent_two_years = current[current.date.gt(pd.Timestamp("2026-09-24") - pd.DateOffset(years=2))]
    counts = {"current_dates": len(current), "current_missing_nav_dates": int((~current.nav_known_in_archive).sum()),
        "current_discount_dates": int(current.premium.lt(0).sum()),
        "current_BASE_static_positive_dates": int(current.BASE_static_positive.sum()),
        "current_STRESS_static_positive_dates": int(current.STRESS_static_positive.sum()),
        "all_available_nav_dates": int(frame.nav_known_in_archive.sum()),
        "recent_two_years_STRESS_static_positive_dates": int(recent_two_years.STRESS_static_positive.sum())}
    source_differences = pd.read_parquet(probe.OUT / "results/source_quote_differences.parquet")
    for row in source_differences.itertuples():
        old_space = probe.space(row.source_quote, row.unit_nav, probe.market_source.distribution.COSTS["STRESS"])
        assert old_space["static_positive"] == row.STRESS_static_positive
    stats = next(row for row in prior["forward_attribution"] if row["period"] == "CURRENT" and row["subset"] == "STRESS_STATIC_POSITIVE")
    early = next(row for row in prior["periods"] if row["period"] == "EARLY")
    result = {"at": now(), "study_id": STUDY, "status": "COMPLETED_DISCOUNT_COMPENSATION_EXTENSION_NO_STRATEGY_PROMOTION",
        "source_study": source.STUDY, "measurement_study": probe.STUDY, **counts,
        "new_market_nav_dates": addition["new_matched_nav_dates"], "public_requests_total": addition["total_public_requests_including_original"],
        "overlap_nav_dates_equal": addition["overlap_rows"], "prior_early_measurement": early,
        "current_event_records": current_events[columns + ["close", "unit_nav", "premium", "STRESS_static_net_return"]].to_dict("records"),
        "unchanged_current_STRESS_attribution": stats,
        "source_quote_corrections_change_STRESS_membership": False,
        "next_action_implication": "当前ETF相对净值价差缺乏频繁且可执行的压力成本后补偿；不据此再筛选分位规则。指数本身的风险补偿与ETF价差分开研究。",
        "account_status": "NOT_RUN_ECONOMIC_COVERAGE_AND_DECISION_TIME_NAV_NOT_ESTABLISHED",
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    record_text = []
    for row in current_events.itertuples():
        record_text.append(f"- {row.date.date()}：相对净值折价{abs(row.premium):.4%}，净值不变并完全修复的压力成本后空间{row.STRESS_static_net_return:.4%}；随后五日收盘到收盘ETF含息收益{row.etf_return_close5:.4%}，其中净值含息变化贡献{row.nav_component_close5:.4%}、相对净值价差变化贡献{row.spread_component_close5:.4%}。")
    report = f"""本轮完成510300相对净值折价的成本补偿测量，并新取得31个日期的单位净值。更新后，2020-01-02至2026-09-24共{counts['current_dates']}个交易日全部有净值，其中{counts['current_discount_dates']}日存在折价，但只有{counts['current_STRESS_static_positive_dates']}日在净值不变、折价完全修复的情景下还能覆盖原压力费用；最近两个日历年仅{counts['recent_two_years_STRESS_static_positive_dates']}日。该证据不足以建立现行高夏普账户，目标仍为active、未完成。

本次检验的机制。

“市场资金紧张”和“能够以足够便宜的价格买入”是不同命题。本轮直接测量ETF价格相对其单位净值的让步，而不再筛选旧研究已做过的份额变化、折溢价分位或融资杠杆阈值。测量只解释ETF这一层价差；它不能排除净值所代表的股票组合本身存在错误定价或风险补偿。

每个已观察收盘报价分别采用10万元预算、100份整手、0.001元价格单位、原BASE/STRESS双边费用，计算买入后在净值不变且完全回归净值时的净空间。10万元仅是20万元账户50%仓位上限下的报价比较尺度，不是已通过ES或回撤预算的真实订单。阈值始终是扣除原费用后空间大于零，没有改分位、成本或窗口。

这个情景没有计入等待期间的市场风险，也没有证明当时能在该收盘价成交；它不是可实现收益，更不是单边策略收益上界。基金净值于收盘后获知时，不能回到该收盘买入。旧数据的feature_asof=15:00只是合并标签，不能证明实际发布时间。

新取得的证据。

历史两段净值在2021-08-12接合，原有3430个日期。新请求范围固定为2026-08-10至2026-09-25，实际两来源均返回截至2026-09-24的34行，其中3行与旧文件重叠且一致，31行是新日期，与已核对的报价逐日匹配。更新后完整历史共有{counts['all_available_nav_dates']}个净值共同日。请求截止9月25日，不意味着供应商已返回9月25日。

本次共3项公开请求：新浪1项、东财2页。东财把首次请求pageSize100限制为实际20行，原两请求方案因此按未完成保存；随后独立完成程序仅补唯一缺页14行，没有重复下载新浪或东财首页。原不完整记录、原净值文件与原研究结果均保留。

两来源相同不等于两个独立官方首次发布记录。本次只有当前请求/收取时间得到记录，历史首版、首次公告和首次送达仍未认证。没有恢复PCF/IOPV计划采集。

空间与收益来源。

2012—2019的1828个共同日中，压力费用后静态空间为正113日；2020年起的{counts['current_dates']}日中仅3日。新增31日没有出现新的压力正空间事件，因此原三个事件及其后续归因逐值不变。BASE成本下当前时期有{counts['current_BASE_static_positive_dates']}日静态正空间，这不能代替现行压力成本门槛。

{chr(10).join(record_text)}

三个日期只能按顺序保留2个不重叠五日区间，不能按三个独立事件或一条策略的三个成功样本解释。2024年10月9日之后，折价收窄贡献为正，但净值下跌更大，ETF整体仍亏损；折价修复不能锁定只持ETF账户的收益。

三个日期的五日ETF含息均值为{stats['outcomes']['etf_return_close5']['mean']:.4%}，20日区块区间为[{stats['outcomes']['etf_return_close5']['lower95']:.4%}, {stats['outcomes']['etf_return_close5']['upper95']:.4%}]。极小样本下，这个区间也不能提供可靠的稳定性认证；2000次抽样中只有{stats['outcomes']['etf_return_close5']['valid_bootstrap_draws']}次包含合格事件。原始逐日与逐事件结果完整保留，不依据较好的平均数晋升。

收益归因采用每份持仓的恒等式：ETF价格变化加现金分红，等于单位净值变化加现金分红，再加ETF价格减净值的价差变化。各项除以同一初始价格；历史最大恒等式误差为{prior['decomposition_identity_max_error']:.3g}。这是事后收盘到收盘归因，不是从净值公告后开始的可执行账户。下一开盘开始的标的五日结果亦已保存，单独标明可用时钟未认证。

原净值文件配套收盘价有3个日期与当前已核对报价相差0.001或0.002元，重算前后都未改变压力正空间事件归属。使用的是单位净值，未将存在拆分口径差异的累计净值混入。

交易范围的含义。

上交所介绍的实时ETF折价套利，包含买入ETF、赎回获得成分股并卖出成分股，还需要覆盖固定和冲击成本。[来源：上海证券交易所]({probe.SSE_URL})。现行仅510300与现金的账户并不实施这组交易，因此不能把收盘折价当作已经锁定的保险费。该说明用于区分收益机制，不代表本项目开展申赎或成分股交易。

本轮没有新增拟合或账户，也没有制作审核包或用户表格。保留旧规则的拒绝结果；不通过变换分位、放宽费用或借早期113个事件重新拟合现行两年模型来救回这条路线。指数股票组合本身的风险补偿仍是另一项待证明的问题。当前市场视图为NO_VIEW，新增独立前向观察0；高夏普目标保持active，未完成。
"""
    (OUT / "本轮研究结论.md").write_text(report, encoding="utf-8")
    for path, sha in input_hashes.items():
        assert digest(ROOT / path) == sha
    completed = {"at": now(), "study_id": STUDY, "status": result["status"], **counts,
        "new_market_nav_dates": 31, "public_requests_total": 3, "new_accounts": 0, "new_fits": 0,
        "goal_status": "active", "goal_achieved": False, "independent_forward_observations": 0,
        "orders_authorized": False, "review_package_created": False}
    save(OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_DIRECT_DISCOUNT_MEASUREMENT_AND_31_NEW_NAV_DATES",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=result["status"],
        latest_etf_discount_compensation_round=completed,
        remaining_research_question="ETF相对净值折价不足以建立现行成本下的高夏普引擎；仍需证明指数股票组合本身的可执行风险补偿及独立验证。",
        goal_metadata_note="本轮完成直接价差补偿测量并取得31个新净值日，属于实质进展；目标未完成。")
    for identifier in [probe.STUDY, source.STUDY, STUDY]:
        if identifier not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(identifier)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=STUDY, latest_integrated_experiment=STUDY,
        current_protocol=(probe.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=(OUT / "本轮研究结论.md").relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=result["status"],
        last_research_result="直接折价测量及31个新净值日完成；当前1633日仅3日静态压力空间为正，现行账户未获晋升，总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    marker = "<!-- ETF_DISCOUNT_COMPENSATION_20260926 -->"
    if marker not in text:
        status_path.write_text(f"{marker}\n\n> 2026-09-26 直接折价补偿测量完成，新增31个双源一致净值日；2020年以来1633日仅3日理论修复可覆盖压力费用。没有产生合格策略，目标ACTIVE；见[本轮结论]({(OUT / '本轮研究结论.md').relative_to(ROOT).as_posix()})。\n\n" + text, encoding="utf-8")
    print("直接折价补偿研究及31日新净值结论已保存，主线保持active、未完成。", flush=True)


if __name__ == "__main__":
    run()
