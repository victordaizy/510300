"""第五轮完整账户复算、中文报告和自包含审阅包。"""
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
STAGE = "510300_direct_policy_utility_v1"
REPORT = ROOT / "reports/research" / STAGE
DELIVERY = ROOT / "deliverables/510300夏普1.2持续研究_第五轮直接仓位学习_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_第五轮直接仓位学习_GPT审阅_20260906.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_saved(root: Path) -> dict:
    report = root / "reports/research" / STAGE
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(report / "metrics.csv")
    maximum, accounts = 0.0, 0
    rows = metrics.to_dict("records")
    rows.append({"cost": "ZERO_COST", "model": "U1_PRIMARY_FULL_SHARPE", **result["zero_cost_primary_diagnostic"]})
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
    assert (training.train_rows >= 500).all()
    for _, part in training.groupby(["fit_origin", "train_window"], dropna=False):
        assert part.training_origin_sha256.nunique() == 1
    for row in json.loads((report / "all_quarter_parameters.json").read_text(encoding="utf-8")):
        theta = np.array(row["theta"])
        assert np.isfinite(theta).all() and abs(theta[0]) <= 4 and (np.abs(theta[1:]) <= 2).all()
    targets = pd.read_parquet(report / "targets.parquet").drop(columns="date").to_numpy(float)
    finite = targets[np.isfinite(targets)]
    assert ((finite >= 0) & (finite <= 1)).all()
    alignment = pd.read_parquet(root / "reports/research/510300_overnight_global_information_v1/source_clock_alignment.parquet")
    valid = alignment.available_at.notna()
    assert (alignment.loc[valid, "available_at"] <= alignment.loc[valid, "decision_time"]).all()
    manifest = json.loads((root / f"config/{STAGE}_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    return {"status": "PASS_SAVED_ACCOUNTS_PARAMETERS_CLOCKS_AND_FROZEN_INPUTS",
            "evaluation_accounts": accounts - 1, "diagnostic_accounts": 1, "quarterly_fits": len(training),
            "usable_fits": int(training.usable.sum()), "converged_fits": int(training.optimizer_success.sum()),
            "maximum_metric_recomputation_error": maximum, "benchmark_daily_parity": "EXACT",
            "same_mature_origins_for_matched_training_windows": True,
            "scope": "必要的结构、时钟、参数与账户数值核对；没有进行安全审计"}


def draw() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, name, color in (("U1_PRIMARY_FULL_SHARPE", "直接夏普主方案", "#187d91"),
                              ("U2_PRICE_SHARPE", "只有价格信息的对照", "#c38732"),
                              ("BUY_HOLD", "买入持有", "#60646b")):
        ledger = pd.read_parquet(REPORT / "evaluation/BASE" / f"{key}_ledger.parquet")
        ax.plot(ledger.date, ledger.equity / 200000, label=name, color=color, linewidth=1.6,
                linestyle="--" if key == "BUY_HOLD" else "-")
    ax.set_title("第五轮：直接学习风险收益，是否改善账户表现", loc="left", fontsize=15, pad=45)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    ax.set_ylabel("完整账户净值（初始为 1）")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.03, "2020-01-02 至 2026-08-14 开盘；基础费用；含分红、等待日和退出费用。\n训练采用连续近似；本图展示完整账户，历史重放不等于独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(DELIVERY / "第五轮直接仓位学习_净值.png", dpi=160)
    plt.close(fig)


def build() -> None:
    assert not DELIVERY.exists() and not ZIP.exists(), "交付已存在，禁止覆盖"
    checked = verify_saved(ROOT)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    from research.direct_policy_utility_v1 import label
    primary = next(row for row in result["primary"] if row["cost"] == "BASE")
    stress = next(row for row in result["primary"] if row["cost"] == "STRESS")
    best = result["post_selected_best_base"]
    DELIVERY.mkdir(parents=True)
    lines = ["# 510300 夏普率 1.2 持续研究：第五轮结果与全部中文规则", "",
             "本轮直接学习风险收益和换仓成本，再将仓位建议放进完整账户中检验。价格组使用三十四个因子，全部信息组使用六十五个因子；八个底层模型和四个共识，共十二个登记候选。所有候选、两种费用和买入持有基准均完整保留。", "",
             f"预先指定主方案基础夏普 **{primary['net_sharpe']:.4f}**，压力夏普 **{stress['net_sharpe']:.4f}**；基础年化收益 **{primary['annualized_return']:.2%}**，最大回撤 **{primary['max_drawdown']:.2%}**。两种费用下主方案是否都达到一点二：**{'历史点估计达到，独立验证待完成' if result['primary_point_target_in_both_costs'] else '未达到'}**。", "",
             f"本轮事后最好为“{label(best['model'])}”，基础夏普 **{best['net_sharpe']:.4f}**，年化收益 **{best['annualized_return']:.2%}**。事后最好不会替换主方案，也不构成独立样本验证。", "",
             f"主方案零费用诊断夏普为 **{result['zero_cost_primary_diagnostic']['net_sharpe']:.4f}**，用于判断摩擦影响。训练近似目标、零费用诊断和实际账户收益分别保留，只有实际账户用于验收。", "",
             f"本轮完成 {checked['quarterly_fits']} 次季度拟合、{checked['evaluation_accounts']} 条完整评价账户和一条诊断账户。可用拟合 {checked['usable_fits']} 次，其中优化器报告收敛 {checked['converged_fits']} 次；达到迭代上限的情况按事前规则记录，不通过重启挑选最优结果。", "",
             f"主方案基础夏普的百分之九十五区块重抽样区间为 {result['uncertainty']['BASE']['primary_sharpe_95_interval']}。这些历史已经被多次研究，区间没有校正全部重复搜索。", "",
             "核对包括保存账户的夏普、年化收益、回撤、现金加持仓加分红应收恒等式，训练样本成熟时钟，跨市场可用时点，参数边界及冻结输入。买入持有的逐日数字与前三轮完全一致。5 项针对性测试已在真实研究前通过。仅做必要的结构和数值核对，未做安全审计；本包没有上传，也没有取得外部 GPT 审阅结论。", "",
             "以下为完整中文因子、训练和交易规则，以及全部候选结果。", "", "---", "",
             (ROOT / "docs/510300_DIRECT_POLICY_UTILITY_V1_PROTOCOL.md").read_text(encoding="utf-8"),
             "", "---", "", (REPORT / "第五轮全部候选_实际结果.md").read_text(encoding="utf-8")]
    (DELIVERY / "第五轮结果与全部中文因子规则.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DELIVERY / "00_先读说明.md").write_text("# 从这里开始\n\n先读《第五轮结果与全部中文因子规则》，再看指标、年度与分期结果、完整逐日账户和季度模型参数。原始数据、分红证据、代码及冻结身份均按原项目相对路径保留。\n\n本包完整交付第五轮及其直接依赖，其他轮次继续保留原包。训练连续近似不等于可交易账户收益，历史达到一点二也不等于独立验证完成。\n\n在所记录依赖环境中，可以运行本脚本的 --verify-only 模式复算保存结果。它不重新训练或下载。没有进行安全审计或外部评审。\n", encoding="utf-8")
    (DELIVERY / "01_GPT审阅提示词.md").write_text("""# 请审阅第五轮直接仓位学习

目标是 510300 与现金的完整账户成本后夏普率一点二，免费来源。请先读中文规则，再核对训练、模型、来源时钟及实际账户。请区分直接优化训练夏普的连续近似、零费用诊断和完整账户结果。

重点检查：训练退出日是否早于拟合，未来数据有没有进入标准化和选参；海外收盘时钟是否早于九点决策；分析梯度、平滑递推和近似换仓费用是否正确；初始半仓是否只用于平滑而没有凭空增加资产；参数上限或优化失败是否如实处理；最低佣金、整手、分红应收、T+1、现金日和终点退出是否完整；主方案有没有被事后最高策略替换；是否误把训练分母的稳定项用进最终夏普。

按证据列出问题、文件位置及复算步骤。判断目标函数变化是否实际改善风险收益，并将费用损耗、预测信息不足和训练与执行近似差异分开解释。不要据一个高训练分数断言未来成功。

提出下一轮一个能实际执行、与本轮实质不同的研究机制。给出中文因子、时间和来源要求、有限候选、训练与验证、完整账户、优先级、验收和停止条件。保留原失败记录，不建议为了凑一点二删费用、挑区间或追加无穷参数。需要额外资产或收费数据时标出依赖，不擅自扩权。新建议是待研究，不是已取得收益，也不授权真实下单。
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
    parser = argparse.ArgumentParser(description="第五轮保存账户复算与自包含交付")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(ROOT), ensure_ascii=False), flush=True)
    else:
        build()
