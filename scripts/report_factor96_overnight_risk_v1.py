"""保存T16固定结果、T06精确边界修正、T13版本台账与本库累计进度。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_overnight_risk_overlay_v1_0_2"
CROWDING = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1"
VERSIONS = ROOT / "reports/research/510300_factor96_supply_version_ledger_v1"
PROGRAM = ROOT / "reports/research/510300_factor96_program_v1"
STUDY = "510300_FACTOR96_OVERNIGHT_RISK_OVERLAY_V1_0_2"
LABELS = {"PRICE_ALL": "价格基准：全部覆盖", "PRICE_COMMON": "价格基准：共同覆盖",
          "D06_ONLY": "仅隔夜下行占比削减", "P02_ONLY": "仅方差低估冲击削减", "FULL": "主方案：任一风险触发削减"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now().astimezone().isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def numeric_reconciliation():
    """只比较保存版本，不重新模拟账户或选择参数。"""
    summaries, comparisons, boundary_rows = [], [], []
    keys = ["period", "capital", "cost", "lag", "policy"]
    for family, old_name, new, primary_lag in [
        ("T06", "510300_factor96_crowding_overlay_v1", CROWDING, 1),
        ("T16", "510300_factor96_overnight_risk_overlay_v1_0_1", OUT, 0),
    ]:
        old = ROOT / "reports/research" / old_name
        pa, pb = read(old / "protocol.json"), read(new / "protocol.json")
        changed_keys = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        assert changed_keys == ["at", "numerical_boundary", "study_id"], changed_keys
        fa, fb = [{r["path"]: r["sha256"] for r in read(p / "freeze.json")["files"]} for p in [old, new]]
        input_names = sorted(k for k in fa if k.startswith("inputs/"))
        assert input_names == sorted(k for k in fb if k.startswith("inputs/"))
        for name in input_names:
            assert fa[name] == fb[name] == digest(old / name) == digest(new / name), name
        before, after = [pd.read_parquet(p / f"daily_features_lag{primary_lag}.parquet") for p in [old, new]]
        shared = [c for c in before.columns if c != "price_long"]
        pd.testing.assert_frame_equal(before[shared], after[shared], check_exact=True)
        changed = before.price_long.ne(after.price_long)
        assert before.loc[changed, "date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-05-29"]
        assert after.loc[changed, "ma20_exact_relation"].eq(0).all()
        changes = []
        for index in before.index[changed]:
            changes.append({"date": str(before.at[index, "date"].date()),
                            "float_wealth": float(before.at[index, "wealth"]),
                            "float_ma20": float(before.at[index, "ma20"]),
                            "prior_long": bool(before.at[index, "price_long"]),
                            "exact_long": bool(after.at[index, "price_long"]),
                            "exact_relation": int(after.at[index, "ma20_exact_relation"])})
        metrics_a, metrics_b = [pd.read_csv(p / "metrics.csv") for p in [old, new]]
        paired = metrics_a.merge(metrics_b, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
        assert len(paired) == 56
        paired.insert(0, "family", family)
        for field in ["net_sharpe", "cagr", "max_drawdown", "end_equity", "net_profit", "fills"]:
            paired[field + "_change"] = paired[field + "_after"] - paired[field + "_before"]
        comparisons.append(paired)
        changed_accounts = 0
        for record in metrics_b.to_dict("records"):
            relative = Path("accounts") / record["period"] / str(record["capital"]) / record["cost"] / f"LAG{record['lag']}" / record["policy"]
            a, b = [pd.read_parquet(p / relative / "decisions.parquet") for p in [old, new]]
            pd.testing.assert_series_equal(a.date, b.date, check_exact=True)
            actions = ["requested_quantity", "filled_quantity", "initial_entry"]
            different = a[actions].ne(b[actions]).any(axis=1)
            changed_accounts += int(different.any())
            row = {"family": family, **{k: record[k] for k in keys}, "changed_action_days": int(different.sum())}
            if record["period"] == "MAIN":
                aa, bb = [z[z.date.eq(pd.Timestamp("2025-05-30"))].iloc[0] for z in [a, b]]
                for field in ["price_long", "initial_entry", "requested_quantity", "filled_quantity"]:
                    row["boundary_" + field + "_before"] = aa[field]
                    row["boundary_" + field + "_after"] = bb[field]
            boundary_rows.append(row)
        summaries.append({"family": family, "prior_root": old.relative_to(ROOT).as_posix(),
                          "current_root": new.relative_to(ROOT).as_posix(), "protocol_changed_keys": changed_keys,
                          "economic_protocol_unchanged": True, "identical_frozen_input_files": len(input_names),
                          "nonpredicate_feature_columns_exactly_identical": len(shared),
                          "price_predicate_changes": changes, "compared_saved_accounts": 56,
                          "accounts_with_changed_action_days": changed_accounts,
                          "prior_implementation_admitted": False, "new_random_draws": 0, "new_accounts": 0})
    pd.concat(comparisons, ignore_index=True).to_csv(OUT / "数值边界修正前后112情景.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boundary_rows).to_csv(OUT / "数值边界交易决策变化.csv", index=False, encoding="utf-8-sig")
    save(OUT / "numeric_boundary_reconciliation.json", {"at": now(), "status": "PASS_SAVED_PRECISION_ONLY_VERSION_RECONCILIATION",
         "studies": summaries, "scope": "只证明经济协议、输入和非判断特征未变，并列保存账户变化；旧实现不再准入，旧文件与旧ZIP保持。"})


def figure():
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.7), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    for policy, color, style in [("PRICE_COMMON", "#8c939e", "--"), ("D06_ONLY", "#b48a37", ":"),
                                  ("P02_ONLY", "#b26c7a", "-"), ("FULL", "#217693", "-")]:
        ledger = pd.read_parquet(OUT / f"accounts/MAIN/200000/STRESS/LAG0/{policy}/ledger.parquet")
        dates = pd.concat([pd.Series([ledger.date.iloc[0] - pd.Timedelta(days=1)]), ledger.date], ignore_index=True)
        nav = np.r_[1., ledger.equity / 200000]
        width = 2.1 if policy == "FULL" else 1.3
        axes[0].plot(dates, nav, label=LABELS[policy], color=color, ls=style, lw=width)
        axes[1].plot(dates, (nav / np.maximum.accumulate(nav) - 1) * 100, color=color, ls=style, lw=width)
    axes[0].set_title("T16 隔夜风险覆盖：联合减仓未改善完整账户收益", fontproperties=font, fontsize=17, loc="left", pad=60)
    axes[0].legend(prop=font, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    axes[0].set_ylabel("完整账户净值", fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）", fontproperties=font)
    axes[1].set_xlabel("2021—2025年；保留全部现金日、费用、分红与期末退出成本准备", fontproperties=font)
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.07, .025, "20万元 · 压力成本 · 精确均线边界修正版 · 风险预算随回撤收缩，低暴露和净值变平不表示盈利恢复", fontproperties=font, fontsize=10, color="#555555")
    fig.tight_layout(rect=[.015, .055, .99, .99])
    fig.savefig(OUT / "主期完整账户净值与回撤.png", dpi=160)
    plt.close(fig)


def main():
    assert not (OUT / "研究结论.md").exists(), "正式报告已存在，不覆盖累计进度"
    result, verification = read(OUT / "result.json"), read(OUT / "saved_verification_receipt.json")
    crowd, crowd_check = read(CROWDING / "result.json"), read(CROWDING / "saved_verification_receipt.json")
    version, version_check = read(VERSIONS / "result.json"), read(VERSIONS / "saved_verification_receipt.json")
    assert verification["status"] == "PASS_SAVED_OVERNIGHT_RISK_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION"
    assert crowd_check["status"] == "PASS_SAVED_CROWDING_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION"
    assert version_check["status"] == "PASS_SAVED_VERSION_ROLE_REFERENCE_AND_ASOF_RECOMPUTATION"
    assert verification["ledgers"] == crowd_check["ledgers"] == 56
    assert not result["historical_joint_point_pass"] and not crowd["historical_joint_point_pass"]
    numeric_reconciliation()
    m = pd.read_csv(OUT / "metrics.csv")
    selected = m.query("period == 'MAIN' and cost == 'STRESS' and lag == 0")
    primary = selected.query("capital == 200000 and policy == 'FULL'").iloc[0]
    small = selected.query("capital == 20000 and policy == 'FULL'").iloc[0]
    early = m.query("period == 'EARLY' and capital == 200000 and cost == 'STRESS' and lag == 0 and policy == 'FULL'").iloc[0]
    delayed = m.query("period == 'MAIN' and capital == 200000 and cost == 'STRESS' and lag == 1 and policy == 'FULL'").iloc[0]
    counts = pd.read_csv(OUT / "overlay_counts.csv")
    main_counts = counts.query("period == 'MAIN' and capital == 200000 and cost == 'STRESS' and lag == 0").set_index("policy")
    tables = []
    for row in selected[selected.capital.eq(200000)].itertuples():
        c = main_counts.loc[row.policy]
        tables.append(f"|{LABELS[row.policy]}|{row.net_sharpe:.6f}|{row.cagr:.3%}|{row.max_drawdown:.3%}|{int(c.cut_episodes)}/{int(c.restore_episodes)}|")
    cost_tables = []
    for row in m.query("period == 'MAIN' and lag == 0 and policy == 'FULL'").itertuples():
        cost_tables.append(f"|{row.capital:,.0f}|{row.cost}|{row.net_sharpe:.6f}|{row.cagr:.3%}|{row.max_drawdown:.3%}|{row.net_profit:,.2f}|")
    increment_tables = []
    for row in result["increments"]:
        other = row["comparison"].removeprefix("FULL_MINUS_")
        increment_tables.append(f"|主方案减{LABELS[other]}|{row['annual_arithmetic_increment']*100:.6f}|[{row['ci95_low']*100:.6f}, {row['ci95_high']*100:.6f}]|")
    decisions = pd.read_parquet(OUT / "accounts/MAIN/200000/STRESS/LAG0/FULL/decisions.parquet")
    ledger = pd.read_parquet(OUT / "accounts/MAIN/200000/STRESS/LAG0/FULL/ledger.parquet")
    cut = decisions.overlay_active & ~decisions.overlay_before
    restore = decisions.overlay_before & ~decisions.overlay_active & ~decisions.base_exit_pending
    decisions.loc[cut].to_csv(OUT / "主方案全部35次削减.csv", index=False, encoding="utf-8-sig")
    decisions.loc[restore].to_csv(OUT / "主方案全部3次恢复.csv", index=False, encoding="utf-8-sig")
    assert cut.sum() == 35 and restore.sum() == 3
    cp = next(r for r in crowd["primary_rows"] if r["capital"] == 200000)
    cs = next(r for r in crowd["primary_rows"] if r["capital"] == 20000)
    old_status = read(OUT / "program_before/status.json")
    current_status = read(PROGRAM / "status.json")
    assert current_status["cumulative_admitted_account_scenarios"] == old_status["cumulative_admitted_account_scenarios"] == 264
    assert current_status["cumulative_executed_account_scenarios"] == 304
    accounting = {"at": now(), "prior_formal": 264, "prior_executed": 304,
        "retired_prior_T06_implementation": 56, "new_T16_invalid_implementation": 56,
        "new_T16_formal": 56, "new_T06_precision_repair_formal": 56,
        "this_round_new_executed": 168, "current_formal": 320, "current_invalid_implementation": 152,
        "current_executed": 472, "invalid_components": {"T02_original": 40, "T06_original_float_boundary": 56, "T16_float_boundary": 56},
        "formal_components": {"T03_T05": 64, "T14": 56, "T02": 40, "T10": 48, "T06_exact_repair": 56, "T16_exact": 56},
        "launch_failure_accounts": 0, "source_only_accounts": 0,
        "identity": "264 - 56 + 56 + 56 = 320；304 + 56 + 56 + 56 = 472；320 + 152 = 472",
        "scope": "情景为时期、本金、费用、对照与滞后组合，绝非320个独立策略或独立样本。"}
    assert sum(accounting["formal_components"].values()) == accounting["current_formal"]
    assert sum(accounting["invalid_components"].values()) == accounting["current_invalid_implementation"]
    save(OUT / "account_count_reconciliation.json", accounting)
    report = f"""# T16隔夜风险覆盖、精确边界修正与T13版本台账

**只操作510300、完整账户成本后夏普至少1.2的目标仍未实现。** T16主方案在2021—2025年、20万元压力成本下净夏普{primary.net_sharpe:.6f}、净年化{primary.cagr:.3%}、最大回撤{primary.max_drawdown:.3%}，期末{primary.end_equity:,.2f}元，累计损益{primary.net_profit:,.2f}元。固定联合风险层触发35次减半、3次恢复，相对同覆盖价格基准的年化收益差为−0.136581个百分点，95%区间跨零。2万元账户同样失败。本轮也修正了会产生一次错误入场的均线浮点边界，T06重算后仍失败；T13只完成32份候选公告的角色和明确版本关系，未计算策略收益。

本库累计完成7个固定研究问题：5个入场候选T03、T05、T14、T02、T10，以及2个持仓覆盖层T06、T16，合格项0。正式账户情景320个，因实现问题退出准入的历史情景152个，累计实际执行472个。其余11项未完成原定义绩效检验。情景数不能当独立策略数，96个因子也未逐项取得独立收益验证。

## 本轮固定问题和基准解释

T16检验的是同一份既有510300持仓在隔夜下行风险占比或方差低估冲击偏高时减半，两个变量同时回落后恢复。D06采用最近60日隔夜负收益平方占两时段平方收益的比例；P02采用当天总收益平方除以此前信息形成的预测方差。它们作为风险覆盖层使用，没有新增独立入场。

附件的“已验证基准”在冻结前明确解释为T06同一MA20、最多5个开盘间隔、完整账本可核对的观察基准；该基准没有通过盈利目标，不能称作已验证的盈利策略。因此本轮能否定这一具体基准上的固定覆盖定义，不能声称完成了在合格盈利基准上的部署验证。没有从旧高夏普账户中换入一个基准，也没有组合失败分支来追求目标。

MA20判断采用包含分红的财富口径。原始报价按0.001元网格转为有理数，判断20倍当日相对财富是否严格大于最近20日相对财富之和；相等时不进入。浮点财富仍保留给图表和风险状态，不决定严格边界的入场。

## 规则、时钟与账户

隔夜收益为(开盘价+当日每份分红)/前收盘价−1，日内收益为(收盘价+分红)/(开盘价+分红)−1，两者复合得到持有日总收益。分红在经济归属日形成应收，付款前不当可交易现金。D06另存60日内最差3个隔夜收益的均值供描述，不增设尾部筛选。

P02方差预测先用严格此前60个完整收益平方的均值初始化，随后按0.94乘前次预测加0.06乘前日收益平方更新。当天平方收益不进入当天分母；中断后重新积累完整60日种子。0.94是计算收益前固定的参数，没有按本轮结果优化。

两变量各自用严格此前252日、至少120个有效值计算90分位和70分位。任一变量严格超过其90分位时，下个合法开盘对旧周期的风险限制后原份额减半，向下取整100份；两个变量都严格低于各自70分位连续两日才恢复风险允许的原份额。中间区间保留削减，未知重置恢复计数并保留削减。单变量对照遵循相应单变量的高低阈值；基准退出始终优先。

主版本在当日收盘计算风险变量，下个开盘仅调整已有周期；首次入场由价格基准和共同可知门决定，不用风险层新开仓。额外滞后一日只是预先固定的敏感性对照。两个主期风险变量全程可知，因此PRICE_ALL和PRICE_COMMON在本轮一致。最多5个开盘间隔的期限适用于全部对照；原周期份额上限只能随风险收缩，不因盈利提高。半仓整手取整到零仍保留旧周期及削减状态，直到原基准退出。

账户仅模拟510300.SH与CASH_CNY，20万元为主、2万元作小资金对照。保留100份整手、0.001元价位、T+1、方向涨跌停、现金限制、分红登记/应收/付款、最低佣金和期末压力退出成本准备。[上交所ETF基础交易说明](https://www.sse.com.cn/assortment/fund/etf/question/)提供股票ETF次日卖出、100份及最小价格变动的交易规则依据；本轮所设滑点与历史成本场景仍属研究假设。

BASE每边万2佣金、最低5元、5bp滑点；STRESS每边万4、最低5元、10bp滑点，成交价向不利方向取整。风险沿用前两自然年、已成熟五日标签的相近状态ES95，每天重估，最高50%名义仓位、2.5% ES预算、−10%跳空下5%损失预算、已有回撤剩余空间折半。达到10%回撤后下一合法开盘退出且不重启。历史最大回撤小于10%不代表真实跳空和延迟卖出时可以保证该上限。

## 完整账户结果和增量

下表均为2021—2025年20万元、压力成本，完整保留1,212个交易日；收益与波动按242日年化，现金和无风险利率按零。252日诊断已另存，没有按结果切换口径。

|固定情景|净夏普|净年化|最大回撤|削减/恢复次数|
|---|---:|---:|---:|---:|
{chr(10).join(tables)}

同一主方案在本金与费用下的完整比较：

|初始本金（元）|成本|净夏普|净年化|最大回撤|累计损益（元）|
|---:|---|---:|---:|---:|---:|
{chr(10).join(cost_tables)}

主期主方案共{int(primary.fills)}次成交、{int(primary.closed_cycles)}个已关闭周期，另有期末未关闭持仓，已计提{primary.terminal_exit_reserve:,.2f}元退出成本准备。佣金{primary.commission:,.2f}元、滑点{primary.slippage:,.2f}元；平均风险暴露{primary.mean_exposure:.3%}，{int(ledger.shares.eq(0).sum())}天收盘零份额。35次削减和3次恢复的全部决策另存CSV。主期有304个统计日满足任一高风险条件，但不一定存在旧持仓或满足持有期限，不能把这304天当304次交易。

2017—2020年20万元压力主方案净夏普{early.net_sharpe:.6f}、净年化{early.cagr:.3%}、最大回撤{early.max_drawdown:.3%}。主期额外延迟一日的20万元压力主方案净夏普{delayed.net_sharpe:.6f}、净年化{delayed.cagr:.3%}，也未达标；不把敏感性版本晋升为主方案。

使用冻结20日循环区块、4,000次已保存抽样，逐日配对比较完整账户收益：

|比较|年化算术收益差（百分点）|95%区间（百分点）|
|---|---:|---:|
{chr(10).join(increment_tables)}

三个区间均包含零，未证明固定联合削减层具有稳定正增量。这些区间未做整个历史研究家族的选择偏差校正；风险预算随各臂净值改变，配对差反映整套动态政策，不是保持完全相同份额的局部因果效应。接近回撤限制后，风险预算逐步收缩，净值变平不代表恢复盈利能力。

![完整账户净值与回撤](主期完整账户净值与回撤.png)

## 数值边界纠正与历史版本保留

独立复算发现2025年5月29日财富指数恰等于20日均值。原浮点值分别为1.8376028501380928和1.8376028501380925，微小正差导致次日多开一次仓位；原始价格与分红的有理数计算证明严格相等。修正前后全样本只改变这一天的价格多头判断，其余共同特征逐值一致，全部直接输入哈希一致，协议仅增加精确比较说明并更新版本号和时间。经济窗口、90/70分位、0.94、恢复规则、成本与风险未变。

T16最初V1因冻结代码路径检查笔误在生成特征和账户前退出，账户数为0；V1.0.1的56个浮点边界账户已否定并原样保留；当前正式版本为V1.0.2。T06原56个账户也退出正式准入，原报告、原ZIP和冻结文件不覆盖，新增V1.0.1仅修复同一严格比较。修正前后112组配对指标及每个账户交易变化均随包保存。这是实现修正，旧策略失败没有撤回，也没有重新搜索经济参数。

T06修正版主期20万元压力净夏普{cp['net_sharpe']:.6f}、净年化{cp['cagr']:.3%}、最大回撤{cp['max_drawdown']:.3%}，亏损{abs(cp['net_profit']):,.2f}元；2万元压力净夏普{cs['net_sharpe']:.6f}。主期仍为7次削减、0次恢复，联合层相对共同价格基准的增量区间跨零，失败结论保持。

账户数清楚分开：原正式264，移出旧T06的56，再纳入修正T06的56和新增T16的56，正式累计320；本轮实际新执行168，累计472；另存152个否定实现情景，包括原T02的40、原T06的56、原T16的56。零账户启动失败和T13来源台账不计收益检验。

## T13公告角色和供给日期版本

上一轮取得固定范围内2,191份PDF、2,190份可检索文本，形成1,177条明确日期数量候选。当前未新增网络采集、收益检验或账户；从已有原文处理32份标题候选，并引用7份相关原始公告。角色分为17份“暂缓授予股份”的本次解禁公告、4份辅助意见、6份发行人更正、3份日期延期、2份补充。标题中的“暂缓授予”不应直接理解为本次解禁日期延期。

原文支持10条明确引用关系，仍未声称全部事件都已去重。海航基础同一批2,249,297,094股的三阶段记录分别指向2020-01-26、2021-01-26，最后变为待承诺事项完成、没有确定新日期。最终未知日期保留为OPEN_ENDED_CONTINGENT_NOT_ZERO_SUPPLY；不能因日期消失，把原潜在供给归零或后填成已知日期。6个按当时可见时间的查询分别保留NO_VIEW、当时计划日或开放式延期，未知不冒充零。

同一公司同一上市日也不保证同一事件：广联达2023-02-08的两份公告分别涉及2020年计划第二期名义233,400股和2021年计划第一期名义88,000股，不能按公司加日期合并；名义解除限售数不能替代实际可上市数量。该反例没有手工回填上轮冻结自动字段。

本轮逐页显示检查了海航延期第1页和两份广联达公告第1页，检查范围和图像另存。保守可用时钟仍是历史研究假设，不证明首次HTTP公开时间；完整本次批次身份、更正冲突、M06全部发行/缴款/上市日历及自由流通市值分母均未齐，T13保持NOT_RUN。T12实际回购用途与自由流通市值门、T04点时权重、T07日频动态体系申赎及T09套息/分红点数门均未因此消失。

## 验证边界、停止条件与后续

T16冻结前19项测试通过；T06精确修正版16项、T13版本台账7项通过，其中精确价格辅助函数的5项测试在T16和T06回执中重复包含，不能简单相加为42项独立测试。独立复核用与生产代码相反方向的有理数递推重算均线比较，分别核对T16和T06各56份账户、61,208行账本和2,990份成熟风险记录。T16另重算时段收益、事前方差、严格历史分位和状态恢复；T06另复算1,330,832条分类交易收益和14,771条官方停牌记录。T13复核覆盖39份原文身份、32角色、10引用、3个日期版本和6个时点查询。

以上属于固定输入到保存结果的算术及逻辑复核，独立前向观察仍为0，外部GPT审阅未进行，未证明历史首版数据在当时已可获得。本轮没有新模型搜索、订单或真实成交，当前市场判断仍为NO_VIEW。交付包结构验证与科学有效性分开，完整冻结与失败记录均保留。

T16这一定义按失败结束，不调整60日、252日、0.94、90/70分位、两日恢复或价格基准，也不把单因子对照晋升为赢家；T06不因精度修正重启参数研究。总目标保持active，后续优先推进可明确区分的T13完整事件身份与M06来源门、T12实际回购披露用途和分母；来源缺口未齐则不计算对应交易特征。T17没有已接受组件，T18仍需成熟校准及重复搜索边界，不拼接历史表现好的局部时段。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    figure()
    save(VERSIONS / "visual_review_receipt.json", {"at": now(), "mode": "模型逐图目视核对",
        "scope": "只复核所列三页，没有OCR回填或改写冻结字段",
        "pages": [{"file": "visual_checks/1209076663_page1.png", "finding": "补偿等承诺尚未完成，延长期限没有明确新日期；保留开放式延期。"},
                  {"file": "visual_checks/1215755330_page1.png", "finding": "2020年计划第二个解除限售期，名义23.34万股，与另一计划分开。"},
                  {"file": "visual_checks/1215755334_page1.png", "finding": "2021年计划第一个解除限售期，名义8.80万股，与另一计划分开。"}],
        "full_event_ledger_verified": False, "trading_feature_admitted": False})
    strategies, factors = read(PROGRAM / "strategy_progress.json"), read(PROGRAM / "factor_progress.json")
    for row in strategies:
        if row["id"] == "T06":
            row.update(current_status="COMPLETE_FIXED_OVERLAY_TARGET_FAILED_EXACT_BASELINE_REPAIRED",
                current_evidence=f"原浮点等值边界56账户已退出准入并保留；精确修正版56账户，主期20万元压力夏普{cp['net_sharpe']:.6f}，7次削减0次恢复，增量区间含0。",
                result_path=CROWDING.relative_to(ROOT).as_posix() + "/result.json")
        elif row["id"] == "T16":
            row.update(current_status="COMPLETE_FIXED_OBSERVATION_BASELINE_OVERLAY_TARGET_FAILED",
                current_evidence=f"同一未获盈利认证的MA20五日观察基准；精确版56账户，主期20万元压力夏普{primary.net_sharpe:.6f}，35次削减3次恢复，增量区间跨0；不构成合格盈利基准上的部署验证。",
                result_path=OUT.relative_to(ROOT).as_posix() + "/result.json")
        elif row["id"] == "T13":
            row.update(current_status="NOT_RUN_EVENT_IDENTITY_ISSUANCE_AND_DENOMINATOR_GATE",
                current_evidence="已有2191份PDF、1177条字段候选；本轮完成32角色、10引用和三阶段延期例证。全部批次去重、M06发行缴款日历和自由流通市值分母未齐。",
                source_gate_path=VERSIONS.relative_to(ROOT).as_posix() + "/result.json")
    for row in factors:
        if row["id"] in ["D06", "P02"]:
            row.update(current_status="FIXED_COMPONENT_TESTED_IN_FAILED_T16_OVERLAY",
                current_note="已在T16固定观察基准检验单项与联合减半覆盖，主方案失败、增量区间跨0；不赋独立因子夏普或声称盈利基准已合格。")
        elif row["id"] in ["F05", "E05"]:
            row["current_note"] += " 本轮原浮点边界修正为精确比较，正式结果转至T06 V1.0.1，失败结论不变。"
        elif row["id"] == "M04":
            row.update(current_status="PARTIAL_VERSION_LEDGER_BUILT_NOT_RUN",
                current_note="1177条字段候选保留；新完成32公告角色、10明确引用及开放式延期实例，尚未全部去重或准入交易特征。")
        elif row["id"] == "M06":
            row.update(current_status="NOT_RUN_ISSUANCE_PAYMENT_AND_FREE_FLOAT_GATE",
                current_note="仍缺全部发行/缴款/上市日历与匹配自由流通市值；M04部分版本台账不能替代M06。")
    save(PROGRAM / "strategy_progress.json", strategies)
    save(PROGRAM / "factor_progress.json", factors)
    pd.DataFrame(strategies).to_csv(PROGRAM / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(PROGRAM / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    status = {**current_status, "at": now(), "completed_fixed_candidates": ["T03", "T05", "T14", "T02", "T10", "T06", "T16"],
        "completed_entry_candidates": ["T03", "T05", "T14", "T02", "T10"], "completed_overlays": ["T06", "T16"],
        "not_run_candidates": 11, "admitted_account_scenarios_this_round": 112,
        "invalid_implementation_accounts_this_round": 56, "retired_prior_admitted_account_scenarios": 56,
        "cumulative_admitted_account_scenarios": 320, "cumulative_invalid_implementation_account_scenarios": 152,
        "cumulative_executed_account_scenarios": 472, "latest_round": STUDY,
        "latest_result": OUT.relative_to(ROOT).as_posix() + "/result.json",
        "new_source_documents_this_round": 0, "new_searchable_text_documents_this_round": 0,
        "source_field_candidates_this_round": 0, "new_version_roles_this_round": 32,
        "new_explicit_version_links_this_round": 10,
        "next_candidates": ["T13_FULL_EVENT_IDENTITY_AND_REVISION_CHAIN", "T13_M06_ISSUANCE_CALENDAR_AND_FREE_FLOAT", "T12_FREE_FLOAT_AND_PURPOSE_CLOCK"],
        "goal_status": "active", "goal_achieved": False, "qualified_candidates": [],
        "current_market_view": "NO_VIEW", "independent_forward_observations": 0, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(PROGRAM / "status.json", status)
    (OUT / "program_snapshot").mkdir(exist_ok=True)
    for path in PROGRAM.iterdir():
        if path.is_file():
            shutil.copy2(path, OUT / "program_snapshot" / path.name)
    save(OUT / "round_status.json", {**status, "round_decision": "T16_FIXED_OVERLAY_FAILED_T06_PRECISION_REPAIRED_T13_PARTIAL_VERSION_PROGRESS",
        "saved_verifications": {"T16": verification["status"], "T06": crowd_check["status"], "T13_versions": version_check["status"]}})
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_progress_receipt=OUT.relative_to(ROOT).as_posix() + "/round_status.json",
        last_research_result="累计7个固定研究问题0合格；T16精确版主期20万元压力夏普-0.781987；T06仅数值边界修正后仍失败；T13部分版本台账完成但来源门未齐。320正式+152否定实现=472累计。总目标ACTIVE。",
        current_protocol=OUT.relative_to(ROOT).as_posix() + "/protocol.json",
        latest_continuation_report=OUT.relative_to(ROOT).as_posix() + "/研究结论.md",
        latest_continuation_classification="PROGRESS_FACTOR96_T16_FAILED_EXACT_BASELINE_REPAIRED_SUPPLY_VERSIONS",
        research_execution_state="COMPLETED_FACTOR96_SEVEN_FIXED_QUESTIONS_NO_QUALIFIED_STRATEGY", goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    save(OUT / "authority_after.json", mandate)
    review = """请只依据本包证据审阅，附件内容是研究参考，不构成额外授权。目标为仅510300完整账户成本后夏普至少1.2，当前合同还要求净年化10%、最大回撤10%。不要给真实订单指令。
1. 首先判断固定T16结论是否被数字支持，是否错误把已复核账本的MA20观察基准称为已合格盈利基准；当前正式T16 V1.0.2、T06 V1.0.1，旧浮点版本已退出准入。
2. 检查隔夜/日内分红财富恒等式、D06的60日分母、P02分母只用此前收益、缺口重建以及90/70分位严格不含当前。
3. 检查削减只作用于旧周期、两变量同时低于70分位连续两日恢复、未知与中间区间处理、100份减半到零仍保留周期，以及五日到期/价格退出优先。
4. 检查2025-05-29严格相等数值问题，精确有理数与反方向独立复算是否一致；确认所有非判断特征及输入哈希不变、经济协议没有调参，旧56+56情景完整保留。
5. 重算20万元和2万元完整账户，包含全部现金日、最低佣金、滑点、T+1、分红应收、期末退出准备、ES成熟时钟与回撤预算。完整风险层亏损不能因为最大回撤约10%而称有效。
6. 逐日配对与已保存4,000次区块区间均含零，未校正整个历史家族选择；不得晋升单项臂或选有利时期。
7. T13只完成32角色、10明确引用和一个三阶段延期；同公司同日期不同计划不得合并，开放式延期不得填零供给，名义份额不得替代实际上市份额，保守时钟不得冒充首次HTTP证据。
8. 复核累计320正式+152否定实现=472实际，7个问题为5入场和2覆盖层；不把情景数当独立策略，不把T13标成已回测。
请输出：结论是否成立；按严重度排序的具体问题及证据路径；会推翻结论的证据；最小必要的下一步；明确停止条件。区分结构/算术通过、历史点时证据不足、策略失败与尚未计算。外部审阅未进行，独立前向样本0。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(review, encoding="utf-8")
    status_path = ROOT / "RESEARCH_STATUS.md"
    old_text = status_path.read_text(encoding="utf-8") if status_path.exists() else ""
    assert STUDY not in old_text
    status_path.write_text(old_text + f"\n\n## {now()} — {STUDY}\n\nT16固定联合风险覆盖失败：20万元压力夏普{primary.net_sharpe:.6f}、年化{primary.cagr:.3%}、回撤{primary.max_drawdown:.3%}，35次削减3次恢复；增量区间跨0。共用价格边界精确修正，旧T06与T16浮点56+56账户退出准入，原包保留；T06修正版仍失败。T13完成32角色、10引用、三阶段延期与6个时点查询，完整事件身份和M06来源/分母未齐，仍NOT_RUN。累计7个固定问题0合格，320正式+152否定实现=472执行。总目标ACTIVE。详见reports/research/510300_factor96_overnight_risk_overlay_v1_0_2/研究结论.md。\n", encoding="utf-8")
    print("T16正式报告、精确边界对账、图表和进度已保存：7个固定问题，320正式情景，0项合格。", flush=True)


if __name__ == "__main__":
    main()
