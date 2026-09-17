"""汇总固定简单核心诊断、更新研究处置并建立自包含结果包。"""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_core_window_diagnostic_v1"
DELIVERY = ROOT / "deliverables/510300简单核心三窗口诊断_20260913"
DELIVERY.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def table(frame):
    lines = ["|规则|年化净收益|净夏普|实际年化波动|最大回撤|平均实际仓位|", "|---|---:|---:|---:|---:|---:|"]
    for _, row in frame.iterrows():
        name = f'{int(row.window)}日简单核心' if row.model == "SIMPLE_CORE" else {"BUY_HOLD": "买入持有", "ETF_VOL10": "ETF单层波动配置"}[row.model]
        lines.append(f"|{name}|{row.annual_return:.2%}|{row.sharpe:.3f}|{row.volatility:.2%}|{row.max_drawdown:.2%}|{row.mean_exposure:.2%}|")
    return "\n".join(lines)


def main():
    protocol = read(OUT / "protocol.json")
    result = read(OUT / "result.json")
    verification = read(OUT / "saved_offline_verification.json")
    assert verification["status"].startswith("PASS")
    metrics = pd.read_csv(OUT / "account_metrics.csv")
    concentration = pd.read_csv(OUT / "cycle_concentration.csv")
    sections = []
    for period, title in [("main", "2020年1月2日至2026年9月11日，1,624个交易日"), ("early", "2015年1月5日至2019年12月31日，1,219个交易日")]:
        for cost, label in [("STRESS", "压力成本"), ("BASE", "基础成本")]:
            frame = metrics[(metrics.period == period) & (metrics.cost == cost)]
            frame = pd.concat([frame[frame.model == "SIMPLE_CORE"].sort_values("window"), frame[frame.model != "SIMPLE_CORE"].sort_values("model")])
            sections.append(f"### {title}／{label}\n\n{table(frame)}")
    concentration_lines = ["|时期／窗口|完成周期|总净利润（元）|最大盈利周期|最大周期占总净利润|前五盈利周期占总净利润|未完成周期盈亏（元）|",
                           "|---|---:|---:|---|---:|---:|---:|"]
    for _, row in concentration[(concentration.cost == "STRESS") & (concentration.category == "primary_accounts")].iterrows():
        label = ("主历史" if row.period == "main" else "较早历史") + f"／{int(row.window)}日"
        concentration_lines.append(f"|{label}|{row.completed_cycles}|{row.total_net_profit:,.2f}|{row.top1_entry}→{row.top1_exit}|{row.top1_share:.2%}|{row.top5_share:.2%}|{row.unfinished_cycle_profit:,.2f}|")
    report = f"""# 510300简单核心三窗口诊断：停止这条路线的扩展优化

本次固定拆解没有找到跨50／60／70日一致、足以支持继续复杂化的核心优势。按用户给定的研发去留原则，停止围绕日内—隔夜信号主导的这条路线追加参数搜索、模型或仓位规则；现版保留为固定研究对照，继续按既定入口留存新数据。原年化10.31%、夏普1.22属于已反复选优历史上的结果，不作为未来预期。

这不是证明日内—隔夜信息在所有用法下都无效，也不是计算出了某个正式的“过拟合概率”。本次没有做新的统计显著性检验、PBO估计、重新训练或独立样本验证；研发停止判断来自已登记的窗口一致性、简单基准比较和利润集中性三项证据。

## 已完成范围与口径

- 三窗口、两成本、两时期，共12份实际模拟主账户，17,058条逐日记录。全部保留，没有挑选最佳窗口。
- 另有12份原价格参考账本、8份比较基准账本。其中主历史4份基准直接复用，较早历史4份为了对齐末日收盘口径重新计算。实际新生成28份账户，复用4份，共32份保存账本、45,488条逐日记录；参考账本和基准不计作新的策略机会。
- 使用已经接纳、截止2026年9月11日的3476行行情因素及14条分红事件。诊断程序行情下载0、新模型拟合0、重采样0、独立前瞻日期0。方法背景另查阅下文所列原论文。
- 初始20万元；每年242交易日；现金及无风险收益率0。基础佣金万2、最低5元、滑点5基点；压力佣金万4、最低5元、滑点10基点。沿用0.001价位、整百份、现金约束、T+1、原方向涨跌停判断、登记／除息／支付分红记账。
- 两个时期均在末日收盘估值，不人工末日开盘清仓；待次日执行的末日决定不计入本期成交。四份较早基准与旧版本前1218日一致，仅对齐终点结算。禁止和旧末日开盘清仓指标混用。

## 怎样保留原退出规则

分数是最近N日“日内对数收益－隔夜对数收益”的和，除以同一N日差值的样本标准差与√N的乘积。N只取50、60、70；最长持有仍为60交易日。连续两日分数严格大于1允许申请入场；连续两日严格小于0、含分红持仓亏损6%、从持仓价值高点回落8%、持有满60日，任一成立即申请次日开盘全部退出。

为严格复用原定义，本次直接调用原价格账户状态机，关闭学习控制器。该参考在一个周期内只买入一次、最终全部退出，产生明确的0／1目标；其含分红成本和高点计算保持原实现，因此波动调仓不会把减仓误判为价格下跌。再入场许可也保持：原价格参考实际清仓后，旧入场条件先归零；同时决策日与实际退出日索引差至少为2，之后才可产生下一开盘申请。

唯一资金账户的当日目标等于该0／1乘以min(1,10%／ETF自身20日年化波动)，没有其他倍率、模型、来源或计划组合。正目标与当前仓位相差不足10个百分点不调仓；明确零目标直接请求清仓，不额外等待两次归零。目标依据收盘生成，按下一开盘尝试固定份额成交。

这里保留的是原价格参考的入场与退出状态，资金账户的实际份额会因波动缩放、带宽、费用和成交约束不同。它不是另行发明“波动调仓后按总持仓市值触发止损”的策略。参考账本只是信号来源，没有从其低波动提取风险放大，也没有把其收益再合成进资金账户。逐日参考和资金决定均在包中。

本次同时移除学习退出、内部低波动放大、1.15倍乘数、多计划合成、额外两次归零等待和辅助机会。因此原版与简单核心的差额不能归因给某一个模块，更不能据此声称学习退出贡献了全部差额。

## 两时期、两成本完整比较

{chr(10).join(sections)}

主历史压力下，50日和60日的年化与夏普高于ETF单层波动配置，70日两项均低于它；相对买入持有，三个窗口年化均更低，只有60日夏普略高。较早历史压力下，三窗口年化都低于ETF单层波动配置，只有60日夏普更高。基础成本没有改变这些比较方向。不存在邻近窗口一致占优的结果。

简单核心的实际平均仓位和年化波动均更低，因此仅凭最大回撤较浅不能认定择时有优势。这里同时展示收益、夏普、波动和仓位；没有追加风险匹配倍率去寻找达标曲线。

## 利润仍然集中

{chr(10).join(concentration_lines)}

最大周期／前五的分母是全账户累计净利润，并包括未完成周期按末日收盘计价的盈亏；比例超过100%表示该盈利已被其他周期亏损抵销，计算没有截断。未完成周期单列，不装成已结束交易。

主历史60日压力总净利润24,946.90元，最大盈利周期为2024年9月25日至10月10日，贡献81.25%的总净利润。预先固定的9月25日至10月8日事件窗口贡献24,314.55元，占总净利润97.47%；之后的回吐使完整周期占比降为81.25%。这两个口径不能混为一谈。

70日同一行情的完整盈利周期贡献677.29%的全期净利润，意味着其他结果抵销了大部分该次盈利；50日没有参加该固定事件窗口，其最大盈利周期换成2020年6月至9月，仍贡献125.23%的累计净利润。问题并未因移除复杂模块而消失。

主历史2024年压力收益分别为50日−0.51%、60日+9.76%、70日+15.79%。不能用这一年70日较好去替换60日；它的全期压力年化只有0.28%、夏普0.075。

## 研发处置与日常记录

1. 现版标记为“高过拟合风险、局部稳健性未通过”，停止围绕10%／1.2修补参数；保留数据、原完整账本、执行引擎及模型快照。
2. 本次固定拆解到此结束。没有选定50／60／70中的任何一条；不进入55／58／62日搜索，不增加过滤器，不以某个基准的低收益为理由提高倍率。
3. 对这条信号主导的复杂化路线停止追加优化预算。若未来要继续研发，应另有事前提出且可被否证的新证据问题，不能把本次较好行自动升级为下一候选。
4. 日常继续留存既定原版的新数据和固定规则决定，定位为研究观察。学习退出只作为原版中的固定观察对象，不为救绩效增加复杂度。新样本可以验证生成时点和执行记录，少量新日收益不能证明长期夏普。
5. 简单核心结果单独存档，不与原版拼接前瞻净值，不替换用户已续填的D60手算Excel；本次没有启用新的每日交易方案。

研究索引的current_strategy_disposition和next_work已写明本处置，优先于历史“最佳候选／点值达标”字段。旧日常任务仍按已有工作日15:30安排；本次不更改其定时配置。该任务原提示要求先读取最新索引并严格按next_work接续。10%／1.2只保留为最终独立验收目标。

## 复核范围与边界

标准库离线程序已复算32份保存账本、45,488行财富恒等式与日盈亏、45,520行决定时点，核对17,070个主账户分数／目标映射，并逐日验证12份价格参考的再入场和退出状态、仓位带宽及主要绩效。复核过程不生成新账户、不拟合、不取行情、不生成随机样本。ZIP解压后的复算回执另随交付提供。

输入行情和分红复用此前接纳证据，没有重新向交易所逐日校验全部3476行，也没有验证真实券商可成交性。本地数学一致性通过，不等于独立绩效验证或外部专家已经认可。本轮账户没有失败重跑；原两个诊断包作为旧证据保留，身份哈希回执随包提供，旧完整大包不再嵌套。

用户所提供外部评审的4499字符正文完整收录在“用户评审原文.md”；其中“外部实际复算12份账户”等是用户转述的外部工作。本次本地复算计数以本报告和saved_offline_verification.json为准。

对多次历史选优应谨慎的依据可参看原作者论文：[Bailey等，《The Probability of Backtest Overfitting》](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。它说明反复测试同一有限历史会增加发现虚假优势的风险；本次没有用论文中的CSCV方法估计本策略的PBO。
"""
    report_path = DELIVERY / "01_简单核心三窗口诊断报告.md"
    report_path.write_text(report, encoding="utf-8")
    disposition = {
        "updated_at": datetime.now().astimezone().isoformat(), "source": "用户提供评审及本次固定拆解的实际保存结果",
        "current_strategy": "SELECTED_MIX_BAND10_SIMPLE2", "status": "HIGH_OVERFITTING_RISK_LOCAL_ROBUSTNESS_FAILED",
        "role": "FROZEN_RESEARCH_CONTROL", "performance_is_future_expectation": False,
        "signal_led_complexity_route": "STOP_ADDITIONAL_OPTIMIZATION_BUDGET",
        "simple_core": "NO_CONSISTENT_THREE_WINDOW_IMPROVEMENT_PROFIT_CONCENTRATION_REMAINS",
        "selected_new_window": None, "new_modules_authorized_by_this_result": False,
        "daily_work": "原固定规则逐日增量及人工D60导出；禁止从历史最佳字段恢复优化；简单核心单独存档不接入每日账户",
        "learning_exit": "FIXED_SECONDARY_OBSERVATION_NO_NEW_COMPLEXITY",
        "final_acceptance_goal_only": "年化10%／净夏普1.2", "goal_achieved": False, "position_impact": 0,
        "report": report_path.relative_to(ROOT).as_posix(), "result": (OUT / "result.json").relative_to(ROOT).as_posix()
    }
    write(OUT / "research_disposition.json", disposition)
    next_document = ROOT / "docs/510300_CURRENT_RESEARCH_DISPOSITION_20260913.md"
    next_document.write_text("# 510300当前研究处置：停止现版及信号主导路线的扩展优化\n\n用户最新评审及已完成固定简单核心诊断覆盖此前继续寻找10%／1.2参数的安排。完整依据见 " + report_path.relative_to(ROOT).as_posix() + "。\n\n现版为高过拟合风险、局部稳健性未通过的固定研究对照。三窗口简单核心没有一致改善，禁止自动选优、加过滤器、加杠杆、再训练来救现版或重复本次诊断。\n\n日常仅按既有15:30安排继续研究观察：先查官方完整日期，仅有新完整收盘才运行 research.new_daily_input_adapter_v1，设置 config/510300_new_daily_input_adapter_runtime_v1.json。新增日线及固定原账户完成后运行 research.daily_manual_signal_v1，不传 --diagnostics；保留全部新日结果、模型时点和决定。保持原算法，不把历史压力年化10.31%／夏普1.22写成预期能力。\n\n新数据处理继续遵守 docs/510300_WAIT_NEW_COMPLETE_DAILY_DATA_20260913.md 的具体接纳、重试和续算方法。无新日线则安静，不计算旧诊断。保持原完整22状态；当前主账户现金不代表全部内部账户现金。\n\n不替换用户续填的Excel，不把简单核心接入每日账户，不拼接策略版本。模型原有确定性月度流程如需保持固定策略一致性仍按原规则执行；不得另行调参、加特征或改变训练规则。研究观察不构成订单授权。\n", encoding="utf-8")
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = read(index_path)
    if not (OUT / "research_index_before_update.json").exists():
        write(OUT / "research_index_before_update.json", index)
    index.update(updated_at=disposition["updated_at"], status="FIXED_CORE_DIAGNOSTIC_COMPLETE_STOP_SIGNAL_LED_OPTIMIZATION_KEEP_DAILY_OBSERVATION",
                 goal_achieved=False, current_strategy_disposition=disposition,
                 research_primary_focus="停止日内—隔夜信号主导路线的扩展优化；现版固定观察与每日数据留存",
                 historical_stability_status="HIGH_OVERFITTING_RISK_LOCAL_ROBUSTNESS_FAILED",
                 latest_fixed_core_diagnostic={"study": protocol["study_id"], "result": disposition["result"], "report": disposition["report"], "status": disposition["simple_core"]})
    entry = {"study": protocol["study_id"], "result": disposition["result"], "status": result["status"], "primary_accounts": 12, "new_price_reference_accounts": 12, "new_comparison_accounts": 4, "reused_comparison_accounts": 4, "new_model_fits": 0}
    if not any(d.get("study") == protocol["study_id"] for d in index["completed_diagnostics"]):
        index["completed_diagnostics"].append(entry)
    index["next_work"].update(status="FIXED_ORIGINAL_RESEARCH_OBSERVATION_ONLY_NO_OPTIMIZATION", focus=disposition["daily_work"],
        source=next_document.relative_to(ROOT).as_posix(), candidate_round=None, registered=False,
        planned_settings=0, planned_new_accounts=0, planned_new_model_fits=0,
        no_strategy_optimization=True, performance_goal_role="FINAL_INDEPENDENT_ACCEPTANCE_ONLY")
    write(index_path, index)
    write(OUT / "research_index_after_update.json", index)
    write(OUT / "execution_activity.json", {"new_primary_accounts": 12, "new_price_references": 12, "new_early_comparators": 4,
        "reused_main_comparators": 4, "total_new_accounts": 28, "new_model_fits": 0, "research_run_failures": [],
        "historical_strategy_parameters_changed": False, "automation_configuration_changed": False,
        "daily_task_instructions_follow_latest_index": True, "standalone_verification": verification["status"],
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__})
    prior = [
        ("deliverables/510300最新策略评审诊断_20260913/510300最新策略评审诊断_GPT复核包.zip", "7ba47e35177c48933951fcf285d89de55b317f205dcc8d1b151baffb2ba14225"),
        ("deliverables/510300_50_60_70日窗口敏感性_20260913/510300_50_60_70日窗口敏感性_GPT复核包.zip", "acc927726aaaf8a44d4105c6b7f162061db1ff0e3d58be5f38c3816f2b9598fd")]
    prior_receipts = []
    for relative, expected in prior:
        path = ROOT / relative
        actual = digest(path.read_bytes())
        assert actual == expected, "旧包身份不同：" + relative
        prior_receipts.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual, "status": "MATCHES_PREVIOUS_DELIVERED_IDENTITY", "nested_in_new_zip": False})
    write(OUT / "prior_package_identities.json", prior_receipts)
    readme = """# 阅读顺序

先读01_简单核心三窗口诊断报告.md，再读02_外部复核提示词.md、数据/protocol.json和数据/用户评审原文.md。

完整12份主账户、12份价格参考及8份基准在数据目录；CSV供直接阅读，Parquet保留原结构，checkpoint保存下一日状态。account_metrics.csv是完整20份资金账户指标；参考账本不当作候选。比较差额、完整盈利周期、年度收益和2024固定事件明细均保留。

程序与直接输入保留工作区相对路径。离线核验只需Python标准库：在解压根目录执行 `python 离线复核.py --data 数据 --receipt 离线复算回执.json`。它不训练、下载或生成账户。仓库目录另含原登记协议、评审来源及完整直接输入，用于需要时从保存输入重放。在单独解压副本内将仓库设为工作目录、安装其中需求，再运行 `python -m research.simple_core_window_diagnostic_v1 run` 及 `python -m research.simple_core_window_diagnostic_v1 summarize`；这是账户重放，不是新增独立研究。交付实际验证的是保存数据离线复算，未在解压目录重新生成账户。

旧两个ZIP已先核对原交付SHA-256；本包仅选取旧结果摘要作背景，旧完整大包不再嵌套。FILE_INDEX.csv是当前包的权威文件索引，自身不递归计算自身哈希。结构核验与解压后离线回执随交付提供，不称为外部专家审核或独立绩效验证。
"""
    prompt = """# 可直接复制给外部评审

请评审这份510300固定简单核心诊断。先以用户提供评审正文和protocol为准，再核对报告与CSV账本；区分用户转述的旧外部复算、本次本地已完成复算和仍未运行的独立验证。

本次12份主账户=3窗口×2成本×2时期；另12份原价格状态参考，不以其账户波动放大仓位；比较基准共8份，其中4份主历史复用，4份早期对齐收盘后计算。不要把32份账本视为32个独立机会。模型拟合0、独立前瞻日期0。两时期均末日收盘。

请重点核查：原价格参考再入场／退出与单层ETF波动资金账户的对应是否符合这次拆解问题；保留参考状态是否造成需进一步限定的解释边界；主账户0目标是否在下一开盘执行而无额外等待；对照费用和末日时钟是否可比；低仓位低波动能否解释回撤优势；盈利是否仍依赖2024特殊行情；停止扩展优化的判断是否与三窗口结果一致。

请明确指出任何证据不足或实现错误。若无错误，不要再将55／58／62日、提高倍率或新过滤器作为达标选拔赛。需要建议时，请区分机制解释、将来可事前固定的验证问题与策略候选；说明研发优先级、最小新增证据和停止条件。多模块同时移除不能归因某一模块，无显著性检验不能称正式证明无效。年化10%／夏普1.2仅为最终验收目标，本包不授权订单或实盘。
"""
    (DELIVERY / "00_README_FIRST.md").write_text(readme, encoding="utf-8")
    (DELIVERY / "02_外部复核提示词.md").write_text(prompt, encoding="utf-8")
    members = {"00_README_FIRST.md": DELIVERY / "00_README_FIRST.md", "01_简单核心三窗口诊断报告.md": report_path,
               "02_外部复核提示词.md": DELIVERY / "02_外部复核提示词.md", "离线复核.py": ROOT / "scripts/verify_510300_simple_core_saved.py"}
    for path in OUT.rglob("*"):
        if path.is_file():
            members["数据/" + path.relative_to(OUT).as_posix()] = path
    source_paths = {ROOT / item["path"] for item in protocol["frozen_files"] if not (ROOT / item["path"]).is_relative_to(OUT)}
    source_paths.update([ROOT / "requirements.txt", next_document, Path(__file__), ROOT / "scripts/verify_510300_simple_core_saved.py"])
    source_paths.update(OUT / name for name in ["protocol.json", "用户评审原文.md", "signal_inputs.csv"])
    for name in ["candidate_dividend_coverage.json", "acceptance_outcome.json", "saved_verification_receipt.json", "result.json", "source_precision_comparison.csv"]:
        source_paths.add(ROOT / "reports/research/510300_post_selection_extension_inputs_v1" / name)
    source_paths.add(ROOT / "reports/research/510300_post_selection_data_feasibility_v1/dividend_candidate_coverage.json")
    for model in ["BUY_HOLD", "ETF_VOL10"]:
        for cost in ["BASE", "STRESS"]:
            source_paths.add(ROOT / "reports/research/510300_strategy_review_diagnostics_v1/counterfactuals" / model / "earlier" / cost / "ledger.parquet")
    queue = [ROOT / "research/simple_core_window_diagnostic_v1.py"]
    visited = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                if module.startswith("research."):
                    dependency = ROOT / (module.replace(".", "/") + ".py")
                    assert dependency.exists(), dependency
                    queue.append(dependency)
    source_paths.update(visited)
    for path in source_paths:
        assert path.is_file(), path
        members["仓库/" + path.relative_to(ROOT).as_posix()] = path
    old_windows = ROOT / "reports/research/510300_session_window_neighborhood_v1"
    for name in ["protocol.json", "portfolio_metrics.csv", "yearly_returns.csv", "early_baseline_clock_check.json", "saved_offline_verification.json", "2024_september_october_positions.csv"]:
        members["旧证据/窗口敏感性/" + name] = old_windows / name
    old_diagnostic = ROOT / "reports/research/510300_strategy_review_diagnostics_v1"
    for name in ["basic_receipt.json", "all_complete_cycles.csv", "paired_exit_summary.csv", "paired_exit_cycles.csv", "saved_economic_verification.json"]:
        members["旧证据/原策略诊断/" + name] = old_diagnostic / name
    index_rows = []
    for name, path in sorted(members.items()):
        data = path.read_bytes()
        index_rows.append({"member": name, "bytes": len(data), "sha256": digest(data)})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["member", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(index_rows)
    index_data = buffer.getvalue().encode("utf-8-sig")
    (DELIVERY / "FILE_INDEX.csv").write_bytes(index_data)
    target = DELIVERY / "510300简单核心三窗口诊断_GPT复核包.zip"
    temporary = target.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("FILE_INDEX.csv", index_data)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(index_rows)+1
        for row in index_rows:
            data = archive.read(row["member"])
            assert len(data) == row["bytes"] and digest(data) == row["sha256"]
    temporary.replace(target)
    receipt = {"status": "PASS_ZIP_CRC_INDEX_MEMBERS_SIZE_SHA256", "zip": str(target), "bytes": target.stat().st_size,
               "sha256": digest(target.read_bytes()), "members": len(index_rows)+1, "indexed_members": len(index_rows),
               "included_research_modules": len(visited), "external_review_performed": False, "fresh_account_replay_from_zip": False}
    write(DELIVERY / "ZIP结构核验回执.json", receipt)
    extracted = DELIVERY / "离线核验解压"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            destination = (extracted / info.filename).resolve()
            assert destination.is_relative_to(extracted.resolve())
        archive.extractall(extracted)
    check = subprocess.run([sys.executable, str(extracted / "离线复核.py"), "--data", str(extracted / "数据"),
                            "--receipt", str(DELIVERY / "ZIP解压后离线经济复算.json")], capture_output=True, text=True, encoding="utf-8")
    (DELIVERY / "ZIP解压后复算输出.txt").write_text(check.stdout+check.stderr, encoding="utf-8")
    assert check.returncode == 0, check.stdout+check.stderr
    print(json.dumps(receipt, ensure_ascii=False))
    print(check.stdout, end="")


if __name__ == "__main__":
    main()
