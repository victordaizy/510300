"""一次性运行已冻结归因；只使用本地现有输入，不训练或搜索。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conditional_score_policy_v1 import metrics, period
from research.csp_v1_static_vs_timing_attribution_v1 import (
    analytic_bounds, audit_ledger, ideal_paths, joint_bootstrap, opening_inputs_valid, simulate_control,
)
from research.intraday_overnight_increment_v1 import (
    digest, normalize_dividends, normalize_prices, now, require, return_metrics, write_json,
)

CONFIG = "config/510300_csp_v1_static_vs_timing_attribution_v1.json"
SOURCE = "research/csp_v1_static_vs_timing_attribution_v1.py"
TEST = "tests/test_csp_v1_static_vs_timing_attribution_v1.py"
PARENT_ZIP = "deliverables/510300_CONDITIONAL_SCORE_POLICY_V1_GPT_REVIEW_20260905.zip"
PARENT_ZIP_SHA256 = "b358b34c29d4ec3d034ffb5112702441283e754ddc73de3d80c852cc04fbf43e"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def event(output: Path, name: str, **values) -> None:
    with (output / "stage_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"created_at": now(), "event": name, **values}, ensure_ascii=False) + "\n")
    print(f"[{now()}] {name}", flush=True)


def freeze(diagnostic: dict, output: Path) -> None:
    require(not (output / "freeze_manifest.json").exists(), "已有冻结，禁止覆盖")
    parent = ROOT / diagnostic["parent_output"]
    prior = read_json(parent / "freeze_manifest.json")
    model_lock = read_json(parent / "model_freeze.json")
    execution = read_json(parent / "execution_receipt.json")
    validation = read_json(output / "pre_freeze_validation.json")
    require(validation["tests_passed"] == 9 and validation["real_market_data_used"] is False, "合成验证回执不符")
    require(validation["test_sha256"] == digest(ROOT / TEST), "测试文件与验证回执不同")
    require(execution["status"] == "COMPLETED", "原研究没有完成回执")
    expected = [*prior["implementation"], *prior["inputs"].values(), *model_lock["development_artifacts"],
                execution["result"], execution["model_freeze"]]
    for item in expected:
        require(digest(ROOT / item["path"]) == item["sha256"], f"原冻结文件变化：{item['path']}")
    require(digest(ROOT / PARENT_ZIP) == PARENT_ZIP_SHA256, "原审阅包身份变化")
    paths = {ROOT / item["path"] for item in expected}
    paths.update(path for path in parent.rglob("*") if path.is_file())
    paths.update(ROOT / rel for rel in [CONFIG, SOURCE, TEST, diagnostic["protocol"], diagnostic["user_text"],
                                       Path(__file__).relative_to(ROOT).as_posix()])
    paths.add(output / "pre_freeze_validation.json")
    files = {path.relative_to(ROOT).as_posix(): identity(path) for path in sorted(paths)}
    manifest = {"study_id": diagnostic["study_id"], "created_at": now(),
                "state": "FROZEN_BEFORE_NEW_DIAGNOSTIC_ACCOUNTS", "evidence_class": diagnostic["evidence_class"],
                "prior_history_already_observed": True, "new_historical_accounts_run": 0,
                "new_fits": 0, "original_freeze_entries_verified": len(expected), "files": files,
                "parent_archive": {"path": PARENT_ZIP, "sha256": PARENT_ZIP_SHA256},
                "external_review_received": "仅用户粘贴的正文；未收到其链接中的ZIP或证据文件"}
    frozen = output / "frozen_sources"
    for relative in [CONFIG, SOURCE, TEST, diagnostic["protocol"], diagnostic["user_text"], Path(__file__).relative_to(ROOT).as_posix()]:
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write_json(output / "freeze_manifest.json", manifest, exclusive=True)
    event(output, "协议代码与依赖冻结完成", frozen_files=len(files), new_historical_accounts=0)


def verify_freeze(output: Path) -> dict:
    manifest = read_json(output / "freeze_manifest.json")
    for relative, item in manifest["files"].items():
        require(identity(ROOT / relative) == item, f"冻结文件发生变化：{relative}")
    return {"status": "PASS_FROZEN_FILES_UNCHANGED", "files": len(manifest["files"]),
            "freeze_manifest_sha256": digest(output / "freeze_manifest.json")}


def load_period(diagnostic: dict, config: dict, output: Path):
    paths = {name: ROOT / relative for name, relative in config["inputs"].items()}
    receipt, coverage = read_json(paths["price_receipt"]), read_json(paths["dividend_coverage"])
    require(receipt["status"] == "PASS" and receipt["canonical_price"]["output_sha256"] == digest(paths["prices"]), "原价格凭证不匹配")
    require(coverage["complete_history_confirmed"] and coverage["distribution_file_sha256"] == digest(paths["dividends"]), "原分红凭证不匹配")
    require(pd.Timestamp(coverage["coverage_end"]) >= pd.Timestamp(diagnostic["end"]), "分红覆盖不足")
    dividends = normalize_dividends(pd.read_csv(paths["dividends"]))
    require(len(dividends) == coverage["event_count"], "分红事件数量变化")
    calendar = pd.read_parquet(paths["calendar"], columns=["trade_date", "is_open"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"])).sort_values()
    require(not dates.duplicated().any(), "日历重复")
    anchor, end = pd.Timestamp(diagnostic["anchor"]), pd.Timestamp(diagnostic["end"])
    expected = dates[(dates >= anchor) & (dates <= end)]
    require(len(expected) == diagnostic["expected_days"] + 1 and expected[1] == pd.Timestamp(diagnostic["start"]), "冻结账户日历不符")
    data = normalize_prices(pd.read_parquet(paths["prices"], filters=[("date", ">=", anchor), ("date", "<=", end)]))
    require(pd.DatetimeIndex(data.date).equals(expected), "行情日期不是完整冻结日历")
    require(set(data.amount_unit) == {"CNY"} and set(data.volume_unit) == {"share"}, "行情单位不符")
    require(np.isfinite(data.amount).all() and (data.amount > 0).all(), "离线成交额质检未通过")
    data["previous_close"] = data.close.shift(1)
    data.loc[0, "previous_close"] = data.close.iloc[0]
    data["dividend"] = data.date.map(dividends.set_index("ex_date").cash_dividend_per_share).fillna(0.0)
    saved = pd.read_parquet(ROOT / diagnostic["parent_output"] / "features_main.parquet")
    saved = saved.loc[(saved.date >= anchor) & (saved.date <= end)].reset_index(drop=True)
    require(pd.DatetimeIndex(saved.date).equals(expected), "原特征有效性掩码日期不符")
    require(np.array_equal(saved.close.to_numpy(), data.close.to_numpy()), "原保存特征与底层收盘不符")
    require(np.allclose(saved.dividend, data.dividend, atol=1e-14, rtol=0), "原分红特征不符")
    data["feature_valid"] = saved.feature_valid.to_numpy(bool)
    for item in dividends.itertuples():
        if anchor <= item.ex_date <= end:
            require(item.record_date in expected and item.ex_date in expected, "分红权益日期缺失")
            require(expected[expected.get_loc(item.ex_date) - 1] == item.record_date, "分红登记日与除息日不连续")
    current = period(data, dividends, diagnostic["start"], diagnostic["end"])
    old_open = (np.isfinite(data.open) & (data.open > 0) & np.isfinite(data.volume) & (data.volume > 0)).to_numpy()
    new_open = data.apply(opening_inputs_valid, axis=1).to_numpy(bool)
    require(np.array_equal(old_open[1:], new_open[1:]), "旧开盘成交量条件影响当前样本；归因对照不可直接比较")
    contract = {"status": "PASS_REQUIRED_DATA_CONTRACT", "anchor": diagnostic["anchor"],
                "start": diagnostic["start"], "end": diagnostic["end"], "days": current.n,
                "no_view_rows": int((~data.feature_valid).sum()), "new_indicators_computed": 0,
                "downloads": 0, "rows_deleted": 0, "old_new_open_eligibility_differences": 0,
                "opening_clock_note": "新判断不使用全日成交量；原样本两种判断相同；未重跑原账户",
                "offline_volume_amount_quality_only": True}
    write_json(output / "data_contract.json", contract, exclusive=True)
    data.to_csv(output / "diagnostic_market_clock.csv", index=False, encoding="utf-8-sig")
    return current


def review_reconciliation(parent: Path, bounds: dict) -> dict:
    old = read_json(parent / "result.json")
    behavior = read_json(parent / "decision_behavior_summary.json")
    models = read_json(parent / "development/selected_models.json")
    rankings = pd.read_csv(parent / "development/configuration_ranking.csv")
    return {"status": "LOCAL_RECONCILIATION_OF_SAVED_EVIDENCE", "analytic_bounds": bounds,
            "parent_state": old["state"], "parent_economics": old["economics"],
            "full_minus_simple_saved_intervals": {cost: old["robust_statistics"]["paired_increments"][f"FULL_MINUS_SIMPLE_{cost}"] for cost in ("BASE", "STRESS")},
            "interpretation_correction": "FULL相对SIMPLE净年化差区间跨零，夏普差与效用差区间不跨零；三种结论分别保留",
            "prior_local_test_receipt": read_json(parent / "pre_freeze_validation.json"),
            "external_test_outcome_reported_in_pasted_text": "34项通过，1项缺少Parquet依赖；外部原始证据文件未收到",
            "external_claims_not_reperformed_here": ["108条验证账户复放", "原特征全字段独立复算", "原区块/HAC重新抽样"],
            "saved_behavior": behavior, "ranking_rows": len(rankings), "ranking_columns": list(rankings),
            "full_frozen_training_statistics": models["FULL"]["training_statistics"],
            "full_frozen_training_objective": models["FULL"]["objective"]}


def write_report(result: dict, output: Path) -> None:
    econ, stats = result["economics"], result["statistics"]["paired_increments"]
    b = result["analytic_bounds"]
    lines = ["# 510300：静态低仓位、机械再平衡与状态择时归因", "",
             "本轮已经完成，终态 **COMPLETED_DIAGNOSTIC_ONLY_NO_PROMOTION**。原V1拒绝和净夏普1.2目标保持。",
             "只新增四条规则模拟账户，参数拟合、优化器、网格及下载均为0；理想统计路径四条，联合区块统计一次。", "",
             "## 账户结果", "", "所有账户均为20万元初始资金，2020-01-02开盘至2026-08-14收盘，1,604日；分红计入、现金零息。",
             f"原截距固定评分 {b['constant_score']:.9f}，固定目标 {b['constant_target']:.7%}。C0初始买入后不调仓，C1每日同分数、同不交易带及执行规则。", "",
             "|成本|账户|净年化|净夏普|年化波动|最大回撤|平均暴露|成交日|末日权益（元）|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cost in ("BASE", "STRESS"):
        for name in ("C0", "C1", "FULL", "BUY_HOLD"):
            item = econ[f"{name}_{cost}"]
            lines.append(f"|{cost}|{name}|{item['annualized_return']:.4%}|{item['net_sharpe']:.4f}|{item['annualized_volatility']:.4%}|{item['max_drawdown']:.4%}|{item['mean_exposure']:.4%}|{item['trade_days']}|{item['ending_equity']:,.2f}|")
    lines += ["", "FULL和BUY_HOLD来自原保存账本；C0/C1是本轮新增。每份保存及新增账本均完成逐日会计核对。", "",
              "## 固定账户下的人民币分解", "",
              "|成本|C0静态净利润|C1−C0机械差额|FULL−C1状态政策差额|FULL总净利润|",
              "|---|---:|---:|---:|---:|"]
    for cost, values in result["wealth_attribution"].items():
        lines.append(f"|{cost}|{values['static_profit']:,.2f}|{values['mechanical_increment']:,.2f}|{values['state_policy_increment']:,.2f}|{values['full_profit']:,.2f}|")
    lines += ["", "上述人民币分量逐日严格相加。状态差额包含状态改变引起的持仓、风险和执行时序差异，不能解释成已识别的纯预测alpha。", "",
              "## 差额与条件95%区间", "",
              "一次20日循环移动区块，2,000次，种子20260906，所有路径共用同一索引。年化差单位为百分点。", "",
              "|比较|净年化差|净年化差95%区间|夏普差|夏普差95%区间|终值差（元）|终值差95%区间（元）|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, item in stats.items():
        cagr, sharpe, wealth = (item[x] for x in ("annualized_return", "net_sharpe", "ending_equity"))
        ci, si, wi = cagr["interval95"], sharpe["interval95"], wealth["interval95"]
        lines.append(f"|{key}|{100*cagr['point']:.4f}|[{100*ci[0]:.4f}, {100*ci[1]:.4f}]|{sharpe['point']:.4f}|[{si[0]:.4f}, {si[1]:.4f}]|{wealth['point']:,.2f}|[{wi[0]:,.2f}, {wi[1]:,.2f}]|")
    lines += ["", "区间条件于冻结模型和已经观察的历史，没有修正历次研究选择或多重比较。效用差、算术年化差及各路径区间均在 statistics.json 和 comparison_intervals.csv。", "",
              "## 同暴露、同波动的理想统计检查", "",
              "用各成本档原BUY_HOLD净日收益乘一个事后固定比例。EXPOSURE匹配FULL全期平均收盘暴露，VOLATILITY匹配FULL净日波动；两者各匹配一个统计量。费用和分红只按原净收益同比例缩放，未计额外调整摩擦，不能作为满足整手/最低佣金合同的策略。", "",
              "|理想路径|乘数k|参考平均暴露|年化波动|净年化|净夏普|",
              "|---|---:|---:|---:|---:|---:|"]
    for key, match in result["ideal_matching"].items():
        value = result["ideal_economics"][key]
        lines.append(f"|{key}|{match['ratio']:.9f}|{match['ideal_mean_reference_exposure']:.4%}|{value['annualized_volatility']:.4%}|{value['annualized_return']:.4%}|{value['net_sharpe']:.4f}|")
    lines += ["", "匹配使用完整历史，bootstrap内固定k，不纳入k估计和整个策略选择的不确定性。乘正比例保持该参考路径的夏普；它能检查资本投入解释，无法消除全部时变风险差异。", "",
              "## 对外部审阅的本地处理", "",
              f"原系数在合法特征矩形包络下，评分位于 [{b['score_bounds'][0]:.6f}, {b['score_bounds'][1]:.6f}]，原始目标位于 [{b['target_bounds'][0]:.6%}, {b['target_bounds'][1]:.6%}]。20分退出不可达，25个百分点加仓上限不能约束该模型。实际暴露仍能因价格漂移超过目标上界。",
              "FULL的7个成交日是首次建仓加6次后续调仓，持有日数为1,604；不等于只持有7天。原FULL−SIMPLE的已保存净年化差区间跨零，但夏普差与效用差区间不跨零，保留此有限正向证据。",
              "本轮开盘调度不使用当日全日成交量。原样本旧/新开盘判断差异为0；原V1源码、输入、模型和结果未改写。日线模拟仍不能证明历史每笔开盘真实可成交。",
              "收到的是外部审阅正文，未收到其sandbox链接中的ZIP或证据。外部环境报告34项测试通过、1项缺依赖；原本地冻结回执记载35项通过，两者分开保留。本轮9项新增合成测试通过。", "",
              "## 本次诊断结论与下一步边界", ""]
    for cost in ("BASE", "STRESS"):
        c0, c1 = econ[f"C0_{cost}"], econ[f"C1_{cost}"]
        delta = stats[f"FULL_{cost}_MINUS_C1_{cost}"]
        ci = delta["annualized_return"]["interval95"]
        identical = result["control_path_equal"][cost]
        lines.append(f"{cost}：C0/C1路径{'完全相同' if identical else '不同'}；C1共{c1['trade_days']}个成交日。FULL相对C1净年化差为{100*delta['annualized_return']['point']:.4f}个百分点，95%区间[{100*ci[0]:.4f}, {100*ci[1]:.4f}]。")
        if identical:
            lines.append("在这个固定截距与10个百分点不交易带的历史样本里，机械再平衡没有产生额外交易，机械归因差额为零；这不代表所有再平衡机制无效。")
    lines += ["", "本次结果只完成规定对照下的归因。C0/C1不晋升为新策略，日线权重、带宽、窗口、种子和gamma/lambda不追加搜索。新的达标策略建模暂停；不存在本次已验证的合格优先候选或备选。", "",
              "研究范围为510300.SH与CASH_CNY。Paper/Shadow、券商、订单和实盘未授权，真实持仓未知，仓位影响0。", "",
              "## 复核入口", "", "- 协议、用户原文和代码见冻结清单及 frozen_sources。",
              "- 新账户见 accounts；原账户见父研究 evaluation；统计理想路径见 ideal。",
              "- parent_account_checks.json / new_account_checks.json：逐日会计、分红资格及交易合同核对。",
              "- bootstrap_indices.npz / bootstrap_samples.npz：唯一一次区块统计索引与全部抽样指标。",
              "- result.json / execution_receipt.json / delivery_verification.json：结果、预算和完成凭证。"]
    (output / "研究报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_paths(ledgers: dict, output: Path, capital: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for col, cost in enumerate(("BASE", "STRESS")):
        for name, color, style in (("FULL", "#bc4a31", "-"), ("C0", "#234c72", "-"), ("C1", "#2a8e82", "--")):
            ledger = ledgers[f"{name}_{cost}"]
            axes[0, col].plot(ledger.date, (ledger.equity / capital - 1) * 100, label=name, color=color, linestyle=style, linewidth=1.6)
        full, c0, c1 = (ledgers[f"{name}_{cost}"] for name in ("FULL", "C0", "C1"))
        axes[1, col].plot(full.date, full.equity - c1.equity, label="FULL−C1 状态政策差额", color="#bc4a31")
        axes[1, col].plot(full.date, c1.equity - c0.equity, label="C1−C0 机械差额", color="#2a8e82", linestyle="--")
        axes[0, col].set_title(f"{cost}：累计净收益（%）")
        axes[1, col].set_title(f"{cost}：人民币差额（元）")
        for row in range(2):
            axes[row, col].legend(loc="upper left")
            axes[row, col].grid(alpha=.2)
            axes[row, col].axhline(0, color="#666666", linewidth=.6)
    fig.suptitle("510300 固定截距归因｜既有历史事后诊断，不晋升策略", fontsize=16)
    fig.savefig(output / "静态底仓与状态政策归因.png", dpi=150)
    plt.close(fig)


def run(diagnostic: dict, output: Path) -> None:
    started = time.perf_counter()
    verify_freeze(output)
    write_json(output / "run_claim.json", {"created_at": now(), "study_id": diagnostic["study_id"],
               "one_historical_run_consumed": True, "maximum_new_accounts": 4}, exclusive=True)
    budget = {"new_parameter_fits": 0, "optimizer_calls": 0, "parameter_grid_evaluations": 0,
              "new_historical_accounts": 0, "ideal_paths": 0, "joint_bootstrap_calls": 0, "new_data_downloads": 0}
    try:
        config = read_json(ROOT / diagnostic["parent_config"])
        parent = ROOT / diagnostic["parent_output"]
        event(output, "开始读取冻结输入与原保存账本", stage="DIAGNOSTIC_ONLY")
        p = load_period(diagnostic, config, output)
        prior = read_json(parent / "result.json")
        require(prior["state"] == "REJECTED_FROZEN_CONDITIONAL_SCORE_POLICY_V1", "原拒绝状态变化")
        models = read_json(parent / "development/selected_models.json")
        bounds = analytic_bounds(models["FULL"]["coefficients"])
        write_json(output / "analytic_bounds.json", bounds, exclusive=True)
        write_json(output / "review_reconciliation.json", review_reconciliation(parent, bounds), exclusive=True)
        ledgers, checks = {}, {}
        for model in ("FULL", "BUY_HOLD", "SIMPLE", "NO_REPAIR"):
            for cost in ("BASE", "STRESS"):
                key = f"{model}_{cost}"
                ledger = pd.read_csv(parent / "evaluation" / f"{key}_ledger.csv", parse_dates=["date", "request_origin"])
                trades = pd.read_csv(parent / "evaluation" / f"{key}_trades.csv", parse_dates=["date", "origin"])
                checks[key] = audit_ledger(p, ledger, trades, config["costs"][cost], config)
                recomputed = metrics(ledger, config)
                for name, value in recomputed.items():
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        require(abs(value - prior["economics"][key][name]) < 1e-7, f"原保存统计不符：{key}/{name}")
                if model in ("FULL", "BUY_HOLD"):
                    ledgers[key] = ledger
        write_json(output / "parent_account_checks.json", checks, exclusive=True)
        event(output, "原八账户与输入门通过，开始唯一四条新增账户", parent_saved_accounts_checked=8)
        (output / "accounts").mkdir()
        new_checks = {}
        for key in diagnostic["new_account_names"]:
            require(budget["new_historical_accounts"] < 4, "新增账户预算耗尽")
            model, cost = key.split("_")
            budget["new_historical_accounts"] += 1
            event(output, "运行冻结归因账户", account=key, account_count=budget["new_historical_accounts"])
            ledger, trades, decisions = simulate_control(p, bounds["constant_score"], model, config["costs"][cost], config)
            ledger["cost"] = cost
            for label, frame in (("ledger", ledger), ("trades", trades), ("decisions", decisions)):
                frame.to_csv(output / "accounts" / f"{key}_{label}.csv", index=False, encoding="utf-8-sig")
            ledger.to_parquet(output / "accounts" / f"{key}_ledger.parquet", index=False)
            new_checks[key] = audit_ledger(p, ledger, trades, config["costs"][cost], config)
            ledgers[key] = ledger
        write_json(output / "new_account_checks.json", new_checks, exclusive=True)
        ideal, matching = ideal_paths(ledgers, config)
        require(set(ideal) == set(diagnostic["ideal_path_names"]), "理想路径超出冻结范围")
        budget["ideal_paths"] = len(ideal)
        (output / "ideal").mkdir()
        for key, frame in ideal.items():
            frame.to_csv(output / "ideal" / f"{key}.csv", index=False, encoding="utf-8-sig")
        write_json(output / "ideal_matching.json", matching, exclusive=True)
        pairs = []
        for cost in ("BASE", "STRESS"):
            pairs.extend([(f"C1_{cost}", f"C0_{cost}"), (f"FULL_{cost}", f"C1_{cost}"),
                          (f"FULL_{cost}", f"C0_{cost}"), (f"FULL_{cost}", f"IDEAL_EXPOSURE_{cost}"),
                          (f"FULL_{cost}", f"IDEAL_VOLATILITY_{cost}")])
        event(output, "新增四账户会计门通过，执行唯一一次联合区块统计")
        budget["joint_bootstrap_calls"] += 1
        statistics, indices, samples = joint_bootstrap({**ledgers, **ideal}, pairs, config, diagnostic)
        np.savez_compressed(output / "bootstrap_indices.npz", indices=indices)
        np.savez_compressed(output / "bootstrap_samples.npz", samples=samples,
                            path_order=np.array(statistics["path_order"]), statistic_order=np.array(statistics["statistic_order"]))
        write_json(output / "statistics.json", statistics, exclusive=True)
        records = []
        for comparison, values in statistics["paired_increments"].items():
            for name, item in values.items():
                records.append({"comparison": comparison, "metric": name, "point": item["point"],
                                "lower95": item["interval95"][0], "upper95": item["interval95"][1]})
        pd.DataFrame(records).to_csv(output / "comparison_intervals.csv", index=False, encoding="utf-8-sig")
        economics = {key: metrics(ledger, config) for key, ledger in ledgers.items()}
        pd.DataFrame.from_dict(economics, orient="index").rename_axis("account").to_csv(output / "economic_comparison.csv", encoding="utf-8-sig")
        attribution, equal = {}, {}
        for cost in ("BASE", "STRESS"):
            full, c0, c1 = (ledgers[f"{name}_{cost}"] for name in ("FULL", "C0", "C1"))
            table = pd.DataFrame({"date": full.date, "static_profit": c0.equity - config["initial_capital"],
                                  "mechanical_increment": c1.equity - c0.equity,
                                  "state_policy_increment": full.equity - c1.equity,
                                  "full_profit": full.equity - config["initial_capital"]})
            residual = table.full_profit - table.static_profit - table.mechanical_increment - table.state_policy_increment
            require(residual.abs().max() < 1e-7, "人民币归因分解不守恒")
            table.to_csv(output / f"wealth_attribution_{cost}.csv", index=False, encoding="utf-8-sig")
            attribution[cost] = {name: float(table[name].iloc[-1]) for name in table.columns if name != "date"}
            attribution[cost]["max_identity_error"] = float(residual.abs().max())
            equal[cost] = bool(np.array_equal(c0[["cash", "shares", "equity"]].to_numpy(), c1[["cash", "shares", "equity"]].to_numpy()))
        ideal_economics = {key: return_metrics(frame.net_return.to_numpy(float), config["annual_days"]) for key, frame in ideal.items()}
        result = {"study_id": diagnostic["study_id"], "created_at": now(), "state": diagnostic["terminal_state"],
                  "stage": "DIAGNOSTIC_ONLY", "evidence_class": diagnostic["evidence_class"],
                  "budget_used": budget, "economics": economics, "analytic_bounds": bounds,
                  "ideal_economics": ideal_economics, "ideal_matching": matching, "statistics": statistics,
                  "wealth_attribution": attribution, "control_path_equal": equal,
                  "parent_rejection_unchanged": True, "high_sharpe_target_unchanged": 1.2,
                  "new_strategy_candidate": None, "position_impact": 0, "paper_shadow_authorized": False,
                  "live_trading_authorized": False, "strategy_search_authorized": False}
        write_json(output / "result.json", result, exclusive=True)
        write_report(result, output)
        plot_paths(ledgers, output, config["initial_capital"])
        verification = verify_freeze(output)
        verification.update({"new_account_checks_passed": 4, "parent_saved_account_checks_passed": 8,
                             "ideal_matching_checks_passed": 4, "budget_used": budget,
                             "original_study_files_unchanged": True, "new_historical_run_count": 1})
        write_json(output / "delivery_verification.json", verification, exclusive=True)
        receipt = {"created_at": now(), "status": "COMPLETED", "study_state": diagnostic["terminal_state"],
                   "elapsed_seconds": time.perf_counter() - started, "budget_used": budget,
                   "result": identity(output / "result.json"), "freeze": identity(output / "freeze_manifest.json"),
                   "historical_run_count": 1, "position_impact": 0}
        write_json(output / "execution_receipt.json", receipt, exclusive=True)
        write_json(output / "status.json", {"state": diagnostic["terminal_state"], "position_impact": 0,
                   "actual_holdings": "UNKNOWN", "next_parameter_search_allowed": False}, exclusive=True)
        event(output, "封卷归因完成，无策略晋升", state=diagnostic["terminal_state"])
    except Exception as exc:
        write_json(output / "failure_receipt.json", {"created_at": now(), "state": "STOP_DIAGNOSTIC_INCONSISTENCY",
                   "error_type": type(exc).__name__, "error": str(exc), "budget_used": budget,
                   "elapsed_seconds": time.perf_counter() - started, "original_inputs_modified": False}, exclusive=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结或一次性运行510300静态底仓与择时归因")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    diagnostic = read_json(ROOT / CONFIG)
    output = ROOT / diagnostic["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        freeze(diagnostic, output)
    else:
        run(diagnostic, output)


if __name__ == "__main__":
    main()
