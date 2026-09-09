"""第六轮日历策略的保存账户核对与自包含中文交付。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = "510300_calendar_liquidity_timing_v1"
REPORT = ROOT / "reports/research" / STAGE
DELIVERY = ROOT / "deliverables/510300夏普1.2持续研究_第六轮日历条件_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_第六轮日历条件_GPT审阅_20260906.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_saved(root: Path) -> dict:
    report = root / "reports/research" / STAGE
    config = json.loads((root / f"config/{STAGE}.json").read_text(encoding="utf-8"))
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(report / "metrics.csv")
    rows = metrics.to_dict("records")
    rows.extend({"cost": "ZERO_COST", "model": key, **value} for key, value in result["zero_cost_diagnostics"].items())
    maximum, accounts = 0.0, 0
    for row in rows:
        branch = "diagnostic" if row["cost"] == "ZERO_COST" else "evaluation"
        folder = report / branch / row["cost"]
        ledger = pd.read_parquet(folder / f"{row['model']}_ledger.parquet")
        returns = ledger.net_return.to_numpy(float)
        assert len(returns) == 1604 and np.isfinite(returns).all() and (returns > -1).all()
        wealth = np.r_[1.0, np.cumprod(1 + returns)]
        sd = float(returns.std(ddof=1))
        values = {"net_sharpe": returns.mean() / sd * np.sqrt(242) if sd > 1e-15 else np.nan,
                  "annualized_return": wealth[-1] ** (242 / len(returns)) - 1,
                  "cumulative_return": wealth[-1] - 1,
                  "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1))}
        for key, value in values.items():
            if pd.isna(value):
                assert pd.isna(row[key])
            else:
                error = abs(float(value) - float(row[key]))
                assert error < 1e-10
                maximum = max(maximum, error)
        assert float((ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable - ledger.equity).abs().max()) < 1e-7
        assert float(ledger.accounting_error.abs().max()) < 1e-6
        assert ledger.date.iloc[0] == pd.Timestamp("2020-01-02") and ledger.date.iloc[-1] == pd.Timestamp("2026-08-14")
        assert ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
        assert (ledger.cash >= -1e-7).all() and (ledger.shares >= 0).all() and (ledger.shares % 100 == 0).all()
        decisions = pd.read_parquet(folder / f"{row['model']}_decisions.parquet")
        invalid = decisions.view == "NO_VIEW"
        assert decisions.loc[invalid, "reference_weight"].isna().all()
        assert (decisions.loc[invalid, "requested_quantity"] == 0).all()
        assert (decisions.decision_time == decisions.execution_date + pd.Timedelta(hours=9)).all()
        if row["model"] == "BUY_HOLD":
            old = pd.read_parquet(root / "reports/research/510300_adaptive_allocation_v1/evaluation" / row["cost"] / "BUY_HOLD_ledger.parquet")
            for col in ("equity", "net_return", "cash", "shares", "dividend_receivable", "commission", "slippage_cost", "dividend_paid"):
                np.testing.assert_array_equal(ledger[col], old[col])
        accounts += 1
    training = pd.read_csv(report / "training_receipts.csv")
    assert (pd.to_datetime(training.last_label_exit) <= pd.to_datetime(training.fit_origin)).all()
    assert (training.train_rows >= 300).all()
    for _, part in training.groupby("fit_origin"):
        assert part.training_origin_sha256.nunique() == 1
    from research.calendar_liquidity_timing_v1 import calendar_features, prepare, CALENDAR_COLUMNS
    calendar = pd.read_parquet(root / config["inputs"]["calendar"])
    features = calendar_features(pd.DatetimeIndex(calendar.loc[calendar.is_open, "trade_date"]), config)
    saved_features = pd.read_parquet(report / "date_only_calendar_features.parquet")
    pd.testing.assert_frame_equal(features, saved_features)
    # 这些固定日期点只核对因子前缀，不重训模型、不读取新增收益。
    for cut in ("2019-12-31", "2020-02-03", "2024-02-19", "2026-08-14"):
        dates = pd.DatetimeIndex(calendar.loc[calendar.is_open & (calendar.trade_date <= cut), "trade_date"])
        prefix = calendar_features(dates, config)
        pd.testing.assert_frame_equal(features.iloc[:len(prefix)].reset_index(drop=True), prefix)
    data, _ = prepare(pd.read_parquet(root / "reports/research/510300_adaptive_allocation_v1/features.parquet"), calendar, config)
    saved = pd.read_parquet(report / "features.parquet")
    pd.testing.assert_frame_equal(data[["date", "execution_date", "feature_valid"] + CALENDAR_COLUMNS], saved[["date", "execution_date", "feature_valid"] + CALENDAR_COLUMNS])
    manifest = json.loads((root / f"config/{STAGE}_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    return {"status": "PASS_SAVED_ACCOUNTS_CAUSAL_CALENDAR_AND_FROZEN_INPUTS",
            "evaluation_accounts": accounts - 2, "diagnostic_accounts": 2, "quarterly_fits": len(training),
            "maximum_metric_recomputation_error": maximum, "benchmark_daily_parity": "EXACT",
            "date_only_feature_recomputation": "EXACT", "future_calendar_removal_prefix_tests": 4,
            "same_mature_training_origins_in_three_groups": True,
            "scope": "必要结构、日历可知时钟和保存账户数值核对；未做安全审计"}


def draw() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for key, name, color in (("K1_PRIMARY_MONTH_START3", "月初三日主方案", "#187d91"),
                              ("K8_FULL_CONSENSUS", "日历与价格共识", "#c38732"),
                              ("BUY_HOLD", "买入持有", "#60646b")):
        ledger = pd.read_parquet(REPORT / "evaluation/BASE" / f"{key}_ledger.parquet")
        wealth = ledger.equity.to_numpy() / 200000
        peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        style = "--" if key == "BUY_HOLD" else "-"
        axes[0].plot(ledger.date, wealth, label=name, color=color, linewidth=1.6, linestyle=style)
        axes[1].plot(ledger.date, (wealth / peaks - 1) * 100, color=color, linewidth=1.3, linestyle=style)
    axes[0].set_title("第六轮：日历条件能否带来可执行收益", loc="left", fontsize=15, pad=45)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    axes[0].set_ylabel("完整账户净值（初始为 1）")
    axes[1].set_ylabel("回撤（%）")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.02, "2020-01-02 至 2026-08-14 开盘；基础费用；含分红、等待日和期末退出。\n月初三日从第一日开盘买入，未计入买入前的隔夜涨幅。历史重放不等于独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(DELIVERY / "第六轮日历条件_净值与回撤.png", dpi=160)
    plt.close(fig)


def build() -> None:
    assert not DELIVERY.exists() and not ZIP.exists(), "交付已存在，禁止覆盖"
    checked = verify_saved(ROOT)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    from research.calendar_liquidity_timing_v1 import label
    base = next(row for row in result["primary"] if row["cost"] == "BASE")
    stress = next(row for row in result["primary"] if row["cost"] == "STRESS")
    best = result["post_selected_best_base"]
    DELIVERY.mkdir(parents=True)
    lines = ["# 510300 夏普率 1.2 持续研究：第六轮结果与完整中文规则", "",
             f"预先指定的月初三日主方案，基础夏普 **{base['net_sharpe']:.4f}**，压力夏普 **{stress['net_sharpe']:.4f}**；基础年化收益 **{base['annualized_return']:.2%}**，最大回撤 **{-base['max_drawdown']:.2%}**。两种费用下是否达到一点二：**{'历史点估计达到，独立验证尚待完成' if result['primary_point_target_in_both_costs'] else '未达到'}**。", "",
             "本轮完成五个日历确定规则、六个预测模型和三个共识，十四个登记候选。三种输入组分别为十八个日历因子、三十四个价格因子和合并后的五十二个因子。", "",
             f"本轮事后最高来自“{label(best['model'])}”，基础夏普 **{best['net_sharpe']:.4f}**，年化收益 **{best['annualized_return']:.2%}**。这项事后比较不替换主方案。", "",
             f"完成 {checked['quarterly_fits']} 次季度拟合、{checked['evaluation_accounts']} 条完整评价账户和两条零费用诊断账户。主方案零费用诊断夏普 **{result['zero_cost_diagnostics']['K1_PRIMARY_MONTH_START3']['net_sharpe']:.4f}**。", "",
             f"主方案基础夏普的百分之九十五区间为 {result['uncertainty']['BASE']['primary_sharpe_95_interval']}。日历加价格共识相对价格共识的年化日均增量区间为 {result['uncertainty']['BASE']['full_vs_price_annualized_arithmetic_increment_95_interval']}。区间未校正全部历次重复研究。", "",
             "全部账户与前五轮同口径：二十万元，二〇二〇年一月二日至二〇二六年八月十四日开盘，一千六百零四个交易日。完整计入分红、现金等待日、整手、T+1、佣金、滑点和退出费用。", "",
             "5 项针对性测试已在真实收益运行前通过。保存账户与指标、日历特征、训练成熟时间及冻结文件身份已复核；删除未来日历后的特征前缀保持一致，买入持有逐日账户与原引擎完全一致。只做必要结构与数值核对，没有安全审计、上传或外部审阅声明。", "",
             "下面保留全部中文因子、模型、规则与候选结果。", "", "---", "",
             (ROOT / "docs/510300_CALENDAR_LIQUIDITY_TIMING_V1_PROTOCOL.md").read_text(encoding="utf-8"),
             "", "---", "", (REPORT / "第六轮日历条件_全部实际结果.md").read_text(encoding="utf-8")]
    (DELIVERY / "第六轮结果与全部中文因子规则.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DELIVERY / "00_先读说明.md").write_text("# 从这里开始\n\n先读《第六轮结果与全部中文因子规则》，再看指标、年度分期结果、逐日账户、日期因子与训练记录。十四个候选和两种费用全部保留。\n\n本包保留第六轮及其直接依赖的原相对路径。旧个股同月收益排序的失败记录仅作为区分证据，不被恢复。日历条件不会获得买入前的隔夜收益，不计算未来实际交易日总数。\n\n具备记录的 Python 依赖后，可运行本脚本的 --verify-only 模式复算保存结果。该模式不重新训练、不下载、不改写研究。没有进行安全审计或外部 GPT 审阅。\n", encoding="utf-8")
    (DELIVERY / "01_GPT审阅提示词.md").write_text("""# 请审阅第六轮日历条件研究

目标为 510300 与现金的完整账户成本后夏普一点二，免费来源。先读中文协议与实际结果，再核对日期因子、预测和所有保存账户。

重点检查：月初序号是否只依赖已经发生的交易日；月末条件是否严格采用自然日而没有读未来交易日总数；长休市计数是否仅用过去；日期因子是否对应实际成交日上午九点；月初和长假后买入是否错误赚取买入前的隔夜跳空；三组模型是否用相同成熟训练起点；因子标准化和季度拟合是否隔离未来收益；分红、整手、T+1、手续费、现金等待和终点退出是否完整；是否把零费用结果或事后最好者替换为主方案。

列出有证据的问题及文件位置，给出复算步骤，解释日历信息相对价格信息有没有稳定增量，区分点估计、未校正重复试验的区间和独立验证。不要因为文献发现其他市场效应就认定510300必然有效。

提出下一轮一个可以执行且实质不同的机制，给出中文因子、免费数据时钟、有限候选、训练和完整账户、优先级、验收与停止条件。已失败方案和窗口保留，不推荐扩大日历窗口扫描、倒转方向、挑年份或删费用来凑到一点二。额外交易资产或收费数据要明确依赖。新建议只写待研究，不冒充已有业绩，不授权真实交易。
""", encoding="utf-8")
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = ROOT / f"config/{STAGE}_manifest.json"
    paths = {ROOT / item["path"] for item in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]}
    paths.update([manifest_path, Path(__file__)])
    paths.update(p for p in REPORT.rglob("*") if p.is_file())
    for cost in ("BASE", "STRESS"):
        paths.add(ROOT / "reports/research/510300_adaptive_allocation_v1/evaluation" / cost / "BUY_HOLD_ledger.parquet")
    for filename in ("research/__init__.py", "scripts/__init__.py", "tests/__init__.py"):
        if (ROOT / filename).is_file():
            paths.add(ROOT / filename)
    coverage = json.loads((ROOT / "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json").read_text(encoding="utf-8"))
    paths.update(ROOT / item["saved_file"] for item in coverage["official_source_snapshots"])
    for source in sorted(paths):
        target = DELIVERY / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    draw()
    assert verify_saved(DELIVERY) == checked
    rows = [{"path": p.relative_to(DELIVERY).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(DELIVERY.rglob("*")) if p.is_file()]
    with (DELIVERY / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in rows:
            archive.write(DELIVERY / row["path"], row["path"])
        archive.write(DELIVERY / "FILE_INDEX.csv", "FILE_INDEX.csv")
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        assert archive.testzip() is None and len(names) == len(set(names))
        assert set(names) == {row["path"] for row in rows} | {"FILE_INDEX.csv"}
        for row in rows:
            payload = archive.read(row["path"])
            assert len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"]
    receipt = {"zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "members": len(names),
               "crc": "PASS", "index": "PASS", "hashes": "PASS", "recomputation": checked,
               "security_audit_performed": False, "external_review_received": False}
    ZIP.with_suffix(".delivery.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第六轮保存结果复算与交付")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(ROOT), ensure_ascii=False), flush=True)
    else:
        build()
