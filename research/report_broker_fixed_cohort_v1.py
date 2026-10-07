"""交付固定组观察图文与长期事实；不变动冻结观察器或原金融账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_fixed_cohort_observation_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend, digest

ROOT, OUT = study.ROOT, study.OUT
RESULT_OUT = OUT / "implementation_v1_0_1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def read(name):
    return study.parent.read(OUT / name)


def save(name, value):
    study.save(name, value)


def fraction(value):
    return f"{value:.2%}" if np.isfinite(value) else "UNKNOWN"


def render(date_labels_only=False):
    if (OUT / "delivery_receipt.json").exists() and not date_labels_only:
        raise RuntimeError("观察交付已经生成，不重复写入。")
    if date_labels_only and (OUT / "delivery_date_label_amendment.json").exists():
        raise RuntimeError("图形数字日期更正已完成，不重复执行。")
    summary = read("implementation_v1_0_1/summary.json")
    context = pd.read_parquet(RESULT_OUT / "results/原R212完成周期_三个固定时点说明性分组.parquet")
    keys = pd.read_parquet(RESULT_OUT / "results/原17关键日_固定组与未知边界.parquet")
    cases = pd.read_parquet(RESULT_OUT / "results/四原案例240行_固定组传播与价量宏观.parquet")
    if summary["stage_events"] != 143 or len(summary["prefix_checks"]) != 17 or len(keys) != 17 or len(cases) != 240:
        raise ValueError("实际完成范围与预定交付不一致。")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    figures = []
    for case_id, d in cases.groupby("original_episode_id", sort=True):
        d = d.sort_values("date")
        fig, axes = plt.subplots(3, 1, figsize=(12.4, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.15, .7]})
        axes[0].plot(d.date, d.etf_close / d.etf_close.iloc[0], color="#454c5b", label="ETF原价收盘 / 案例首日")
        for name, label, color in (("leaders", "事前价格领先组", "#247568"), ("followers", "其余已知成员", "#ba7836")):
            axes[1].plot(d.date, 100 * d[f"{name}_positive_lower"], color=color, label=label + "累计上涨下界")
            axes[1].fill_between(d.date, 100 * d[f"{name}_positive_lower"], 100 * d[f"{name}_positive_upper"], color=color, alpha=.13)
            ratio = d[f"{name}_known"] / d[f"{name}_size"].replace(0, np.nan)
            axes[2].plot(d.date, 100 * ratio, color=color, label=label + "已知覆盖")
        if d.leaders_size.eq(0).all():
            message = f"NO_VIEW：锚点可排序{int(d.anchor_eligible_count.iloc[0])}/300只，未取得固定组资格\n保留未知，不能据空白判断没有扩散"
            for axis in axes[1:]:
                axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes, fontsize=11, color="#656565")
        axes[1].axhline(50, color="#999999", linewidth=.8, linestyle="--")
        axes[2].axhline(98, color="#999999", linewidth=.8, linestyle="--")
        for axis in axes:
            axis.legend(loc="best", fontsize=9)
            axis.grid(axis="y", alpha=.22)
        axes[0].set_ylabel("ETF价格相对值")
        axes[1].set_ylabel("固定组累计上涨 / %")
        axes[2].set_ylabel("已知覆盖 / %")
        axes[1].set_ylim(-3, 103)
        axes[2].set_ylim(-3, 103)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.suptitle(f"原案例{int(case_id)} · {d.date.iloc[0].date()}起固定价格组\n成分滞后一完整交易日；阴影为未知上下界；低于覆盖门保留NO_VIEW", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .93))
        path = OUT / "figures" / f"原案例{int(case_id)}_固定组观察_交付.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(study.relative(path))
    lines = ["# 固定价格领先组与其余成员传播：全部观察及下一实验", "",
        "2026-10-05，TECH.R213—R214。登记后观察143个原阶段事件、3003条0至20观察；保留3488槽、2855评价日、原四案例240行及17关键日。119个事件取得锚点资格、24个NO_VIEW，143事件包含71个原R212完成周期、1未完成及71无实际周期。", "",
        "结论：接受固定组传播的描述与时序事实，不接受‘两组共同上涨足以筛出高胜率进入’。当前没有新增金融策略或账户，收益夏普NOT_COMPUTED；最新金融仍R212四门0/4拒绝。", "",
        "## 具体上涨段与未知", "",
        "2024：在8月29日固定价格组的原案例中，9月24日观察仍只可用9月23日源，两组累计上涨下界约35.81%/36.91%，未共同过半；9月26日可用9月25日源，约67.57%/72.48%，共同扩散已经形成。9月30日用9月27日源，约84.46%/97.32%。宏观和ETF价格信号与成分统计的时钟不同，不能把后来扩散倒填到原9月24日进入判断。", "",
        "在9月24日原REPRICING事件前一天另固定组，9月25日收盘观察9月24日源时，两组上涨下界约93.92%/100%。该确认比原ETF决定晚一个收盘，若另作进入政策，最早还需下一开盘执行，必须配延迟对照。两个锚点回答不同问题，名单和比较基点不能混用。", "",
        "2019早期修复：1月8日两组约29.05%/38.93%，尚未共同过半；后续扩散逐步形成。2020恢复段的案例锚点位于3月下跌之前，至6月仍有大量成员未回到锚点价以上，7月才共同过半；它不等于3月反弹没有发生，说明累计基点的含义须固定。2015案例锚点的20日可排序覆盖不足，整组为NO_VIEW，不能据此断言扩散弱或用它支持熊市识别。", "",
        "| 原关键日 | 固定组源日 | 当前可用源日 | 领先组上涨下界 | 其余组上涨下界 | 描述状态 |", "|---|---|---|---:|---:|---|"]
    for row in keys.itertuples(index=False):
        lines.append(f"| {row.date.date()} | {row.cohort_source_date.date()} | {row.observation_source_date.date()} | {fraction(row.leaders_positive_lower)} | {fraction(row.followers_positive_lower)} | {row.descriptive_propagation_state} |")
    lines += ["", "局部比例只是已知组的下界。NO_VIEW的已有局部值不会取得策略输入资格；全部未知边界及未排序成员保存在原表。", "",
        "## 全部原实际周期说明性分组", "",
        "这张表引用已经已知的原R212压力完成周期结果，保持其原成交，不是按新规则成交的账户，也不是新的入场胜率。原观察时点1、5、20全部报告；晚于原退出的信息单列，不能回写。", "",
        "| 时期 | 观察相对日 | 固定组状态 | 原完成周期 | 原赢/亏 | 信息在原退出之前的周期 |", "|---|---:|---|---:|---:|---:|"]
    for row in context.itertuples(index=False):
        lines.append(f"| {row.period} | {row.relative_session} | {row.state} | {row.original_completed_cycles} | {row.original_wins}/{row.original_losses} | {row.information_before_original_exit} |")
    lines += ["", "近期首次可观察两组共同过半的29个原周期，最终9胜20负，原净损益合计−7209.98元；5日共同过半也有25个原周期、11胜14负。共同扩散真实存在，却不能充分区分后续成功。较早首次共同过半8个、5胜3负，不具跨时期稳定性。", "",
        "近期5日两组均未过半的13个原周期最终均亏，信息在其中10个原退出之前，3个已经退出；早期同组只有1个亏损且信息晚于退出。20日均未过半的15个近期原周期也全亏，但信息无一早于原退出。故20日结果含显著结果已经发生的解释，不能用来包装提前能力。", "",
        "累计相对锚点上涨也会在回撤后持续为正：2024快速抬升基点后，两组多数仍高于事件前价，不能由该比例单独决定顶部或保护利润。长窗还因累计缺失增加NO_VIEW，240案例142行NO_VIEW、3003事件观察809行NO_VIEW，不能删掉缺口来美化结论。", "",
        "## 接受、拒绝及下一具体实验", "",
        "接受：当时可知的固定身份、扩散顺序、缺失边界和宏观/价格的不同信息时钟。拒绝：把共同上涨直接设成通用进入门、把累计上涨单独用于顶部判断、把第20日已经发生的损失视为提前预测，或把已知原周期分组当新净pB。旧多数、速度、中位与T04状态均保持。", "",
        "下一优先假设为持有初期失效：价格信号已发生后，固定两组持续缺乏收益响应，是否能更早确认失败并减小无效持有。需另登记主政策、同期ETF自身价格失效对照及原退出对照；按真实进入重新固定名单，未知不自动退出，明确统计滞后和次开成交。5日是本轮结果已知的开发选择，不能声称盲测；全部原时期和费用、资金风险、净pB>1、完整收益/夏普与独立要求不变。当前仅假设，尚无已准入待跑金融候选。", "",
        "五项原必要测试及日期源回归共6通过，17前缀身份/观察精确不变，冻结来源哈希保持。首个观察因datetime.date与Timestamp比较错误终止；原冻结代码保持，隔离v1_0_1只规范日期类型，完整观察一次成功，没有改规则或来源。原登记前括号错误和终止失败均保存。四交付图重新排日期轴并明确2015未知，原图保留。", "",
        "历史开发/描述，首版NOT_CERTIFIED、独立NOT_ESTABLISHED、全项目搜索选择校正NOT_COMPUTED；目标active且未实现。", "",
        "[登记协议](protocol.json) · [纯日期类型纠错](implementation_v1_0_1/implementation_amendment.json) · [完整观察结果](implementation_v1_0_1/summary.json) · [全部事件逐日记录](implementation_v1_0_1/results/全部事件0至20观察_固定组传播及未知.csv) · [原周期说明性分组](implementation_v1_0_1/results/原R212完成周期_三个固定时点说明性分组.csv)"]
    for path in figures:
        lines += ["", f"![固定组观察](figures/{Path(path).name})"]
    (OUT / "固定组传播_全部结果与下一实验.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt_name = "delivery_date_label_amendment.json" if date_labels_only else "delivery_receipt.json"
    save(receipt_name, {"at": study.original.now(), "status": "OBSERVATION_REPORT_AND_FIGURES_WRITTEN",
        "result": study.relative(RESULT_OUT / "summary.json"), "figures": figures, "new_accounts": 0,
        "new_fits": 0, "financial_metrics": "NOT_COMPUTED", "goal_achieved": False,
        "visual_only_date_label_amendment": date_labels_only})
    print("固定组观察报告与4张交付图已生成；未新增账户或改冻结观察规则。", flush=True)


def archive():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R214.json").exists():
        raise RuntimeError("本观察已归档或开始，不重复追加。")
    summary = read("implementation_v1_0_1/summary.json")
    viewed = read("figure_delivery_view_receipt.json")
    service = read("goal_service_status_after_observation.json")
    if viewed["viewed"] != 4 or service["goal"]["status"] != "active" or len(summary["prefix_checks"]) != 17:
        raise ValueError("观察、图形或目标实际状态未完成。")
    old_state = STATE.read_bytes()
    state = json.loads(old_state.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R212" or state["latest_actual_financial_decision"] != "TECH.R212":
        raise ValueError("其他研究已经更新本线状态，先核实。")
    forward = {key: state[key] for key in FORWARD}
    report = study.relative(OUT / "固定组传播_全部结果与下一实验.md")
    result = study.relative(RESULT_OUT / "summary.json")
    note = f"""> 最新技术观察事实（2026-10-05，TECH.R213—R214；金融仍R212）：固定信号前价格领先/其余成员，143原事件/3003观察、119合格锚/24未知，全部3488槽/2855评价日、原240案例/17关键日；17前缀身份与观察精确保持、原5必要及日期回归共6测试通过，4交付图已查看。接受扩散和时钟描述，不接受共同上涨足以筛选进入或累计上涨足以判断顶部。近期首次共同过半29原周期9胜20负；5日共同过半25原周期11胜14负；均为已知原R212成交上下文，不是新交易胜率。0新账户/拟合/标签/市场请求，新收益夏普NOT_COMPUTED；最近完整金融仍R212四门0/4及稳定拒绝。目标服务实际active，本轮PROGRESS，连续受阻0，完整目标/独立/去过拟合未达。

具体上涨：2024原案例的8月29日固定组，9月24日观察只可用9月23日源，上涨下界35.81%/36.91%，9月26日用9月25日源67.57%/72.48%；9月24事件前另固定组，9月25收盘才可观察到9月24源93.92%/100%，不能用于原9月25开盘。2019早期与2020恢复仍受锚点基数影响，2015案例20日排序覆盖不够为NO_VIEW。240案例142未知行/3003事件809未知行保留；不降覆盖或改锚点救援。

下一实验假设：持有初期两组没有响应可否与ETF自身价格共同确认失效，配原退出和同期价格失效对照。近期5日两组均未过半13原周期全亏，但只有10周期信息早于原退出、3已经退出；20日同组15全亏但全部晚于原退出，不存在提前资格证明。该5日是结果已知的开发选择，须另写完整金融用途与滞后/未知/真实进入定组，不称盲测或独立。当前无已准入待跑金融；旧多数/速度/中位失败、原T04权重门、原R212配置及13前瞻保持。

实现边界：原冻结观察第一次因日期类型比较错误终止，原版本保留；v1_0_1只规范日期类型，相同源、组、窗口/覆盖/统计时间，完整观察成功一次。未重复旧金融；代码/源哈希保持。首版NOT_CERTIFIED、独立NOT_ESTABLISHED、全项目历史选择校正NOT_COMPUTED。

依据：[全部结果和下一用途](../{report})；[固定观察卡](510300_BROKER_FIXED_COHORT_PROPAGATION_V1.md)；[实际观察及17前缀](../{result})。
"""
    decisions = f"""### TECH.R213—R214：固定价格组传播、提前性与持有失效假设（2026-10-05）

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 信号前固定价格组 | 领先股票和其余成员的后续响应可解释上涨的扩散 | 143原事件，按前一源日20回报分两组、固定身份；3003观察，17前缀 | 119合格锚/24未知，身份与观察截断精确；240原案例/17日期及缺口保留 | 接受描述，非权重/行业指数收益或实际资金流；不声称完成原T04 | 可另立金融用途，须真实进入定组、时钟及独立证据 |
| 共同上涨直接筛进入 | 两组共同累计上涨意味着更高成功率 | 原R212完成周期在1/5/20固定观察时点全部说明性分组 | 近期首次共同29原周期9胜20负，5日25个11胜14负；较早首次8个5胜3负 | 不接受充分筛选假设，原成交上下文非新策略胜率；时期不稳定且确认有延迟 | 不重跑旧广度；不同完整政策须延迟及来源对照 |
| 共同无响应作持有失效 | 信号后两组仍无收益响应可能提示机会失效 | 固定第5观察日，核信息是否早于原退出；未按新规则交易 | 近期13原周期全亏，10早于退出/3已退出；较早1亏但已退出 | 接受下一开发假设，未证明可执行收益；选择5日的结果已经已知 | 是，原退出/同期ETF价格失效对照、相同风险费用全账户和独立验证 |
| 20日弱势是提前预测 | 第20日两组未过半可提前排除亏损 | 全部既有退出日期与信息时间匹配 | 近期15原周期全亏，但信息全部晚于原退出 | 拒绝提前资格与入场用途，结果已发生的描述不能回填 | 仅当信息真正提前且另立完整用途 |
| 累计比例作顶部判断 | 高累计上涨比例代表趋势仍安全 | 原2024完整时序及未知边界 | 抬升后多数仍高于锚点价，回撤时比例仍高；缺失随累计增加 | 不接受单独顶/退势判断，基点不同导致含义不同 | 不调锚点救援；新机制须独立定义和全账户 |
| 旧广度与本定义的关系 | 新定义可以绕过全部旧失败和权重门 | 实际读取E01/E03/T06/T04旧用途，保留原结果；按价格排序另立描述 | 旧多数/速度/中位未达目标，T04仍未跑/权重源不合格 | 拒绝改名复活旧规则；接受另立固定价格组描述，非完成原权重策略 | 只有不同完整用途才继续金融，不套原115模型训练门给无拟合观察 |

6必要/回归测试、17前缀、4图已查看。首个冻结观察日期类型错误终止；原代码不改，隔离v1_0_1纯实现纠错，参数和源均不改。原240案例142未知、事件3003观察809未知保留。0新金融/拟合/标签/市场请求，新收益夏普NOT_COMPUTED，金融仍R212拒绝；目标active、PROGRESS、受阻0，13前瞻与旧原策略/失败保持。独立/去过拟合未建立。

依据：[完整图文](../{report})、[原固定协议](510300_BROKER_FIXED_COHORT_PROPAGATION_V1.md)、[全部观察](../{result})。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        prepared.append((path, *prepared_prepend(path, text)))
    state["previous_phase_before_TECH_R213_R214"] = {key: state.get(key) for key in (
        "latest_technical_decision", "latest_registration_decision", "latest_result", "latest_report", "current_phase", "current_phase_trial_accounting")}
    state.update({"updated_at": study.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_technical_decision": "TECH.R214", "latest_registration_decision": "TECH.R213", "latest_result": result,
        "latest_report": report, "latest_completed_description_study": study.relative(OUT),
        "latest_goal_tool_status_receipt": study.relative(OUT / "goal_service_status_after_observation.json"),
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": study.original.now(),
        "previous_goal_turn_classification": "PROGRESS_R211_R212_TWELVE_NEW_FINANCIAL_ACCOUNTS_AND_SOURCE_DIAGNOSIS",
        "goal_turn_classification": "PROGRESS_R213_R214_FIXED_COHORT_NUMERICAL_DESCRIPTION_AND_TIMING_FALSIFICATION",
        "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "current_phase": "FIXED_COHORT_DESCRIPTION_COMPLETED_NEXT_HOLD_FAILURE_HYPOTHESIS",
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 6,
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "latest_progress": "143原事件/3003观察、240案例/17日期与17前缀；共同上涨不能充分区分进入，5日共同无响应有持有失效假设但需真实账户与时钟。",
        "next_research_question": "固定组无响应能否作真实进入后的失效，与同期ETF自身价格失效和原退出比较；5日是结果已知的开发选择，未知不自动退出，保留统计延迟、同风险费用和独立要求。",
        "next_fixed_cohort_financial_hypothesis": {"status": "HYPOTHESIS_ONLY_NOT_ADMITTED_OR_FULL_FINANCIAL_REGISTRATION",
            "purpose": "HOLDING_FAILURE_CONFIRMATION_NOT_SIMPLE_POSITIVE_ENTRY_GATE",
            "required_controls": ["原退出", "同期ETF自身价格失效"], "new_financial_run": "NOT_RUN"},
        "current_phase_trial_accounting": {"scope": "TECH_R213_R214_FIXED_PRICE_COHORT_DESCRIPTION",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
            "event_anchors": 143, "event_profile_rows": 3003, "case_rows": 240, "key_rows": 17,
            "historical_prefix_checks": 17, "original_tests": 5, "implementation_regressions": 1,
            "completed_observation_runs": 1, "terminal_type_failure_attempts": 1, "delivery_figures_viewed": 4},
        "current_goal_turn_actual_work": {"new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
            "new_fixed_cohort_event_observation_rows": 3003, "timing_checks": 17, "unknown_event_rows_retained": 809}})
    with (OUT / "state_before_TECH_R214.json").open("xb") as stream:
        stream.write(old_state)
    receipts = []
    for path, old, new, body in prepared:
        with (OUT / f"{path.stem}_before_TECH_R214.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if path.read_bytes() != new or not new.endswith(body):
            raise ValueError("原事实正文没有精确保留。")
        receipts.append({"path": study.relative(path), "old_body_preserved_exact": True, "old_sha256": digest(old), "new_sha256": digest(new)})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if {key: study.parent.read(STATE)[key] for key in FORWARD} != forward:
        raise ValueError("原13前瞻字段改变。")
    save("project_state_update_receipt.json", {"at": study.original.now(), "status": "R214_DESCRIPTION_AND_TIMING_LIMITS_SAVED",
        "documents": receipts, "forward_fields_preserved_exact": list(FORWARD), "latest_financial_decision": "TECH.R212",
        "new_financial_metrics": "NOT_COMPUTED", "current_goal_status": "active", "goal_achieved": False,
        "new_accounts": 0, "current_blocked_count": 0})
    print("R214描述、提前性限制及下一持有失效假设已写入4事实文档；金融仍R212、13前瞻保持、目标active未达。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定组观察交付和事实归档。")
    parser.add_argument("action", choices=["render", "amend-date-labels", "archive"])
    args = parser.parse_args()
    if args.action == "archive":
        archive()
    else:
        render(date_labels_only=args.action == "amend-date-labels")
