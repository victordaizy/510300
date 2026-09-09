"""生成三轮研究的中文说明、净值图及自包含审阅包；只做结构与数值检查。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "deliverables/510300夏普1.2持续研究_前三轮_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_前三轮_GPT审阅_20260906.zip"
STAGES = ["510300_adaptive_allocation_v1", "510300_return_classification_v1", "510300_overnight_global_information_v1"]
PROTOCOLS = ["510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md", "510300_RETURN_CLASSIFICATION_V1_PROTOCOL.md",
             "510300_OVERNIGHT_GLOBAL_INFORMATION_V1_PROTOCOL.md"]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_saved(root: Path) -> dict:
    account_count, training_count, max_error = 0, 0, 0.0
    for stage in STAGES:
        folder = root / "reports/research" / stage
        metrics = pd.read_csv(folder / "metrics.csv")
        for row in metrics.itertuples():
            ledger = pd.read_parquet(folder / "evaluation" / row.cost / f"{row.model}_ledger.parquet")
            values = ledger.net_return.to_numpy(float)
            assert len(values) == 1604 and np.isfinite(values).all() and (values > -1).all()
            wealth = np.r_[1.0, np.cumprod(1 + values)]
            sd = values.std(ddof=1)
            calculated = {"net_sharpe": values.mean() / sd * np.sqrt(242) if sd > 1e-15 else np.nan,
                          "annualized_return": wealth[-1] ** (242 / len(values)) - 1,
                          "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
                          "cumulative_return": wealth[-1] - 1}
            for name, value in calculated.items():
                original = getattr(row, name)
                if pd.isna(value):
                    assert pd.isna(original)
                else:
                    error = abs(float(original) - value)
                    assert error < 1e-10, (stage, row.cost, row.model, name, error)
                    max_error = max(max_error, error)
            recomputed_nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
            assert float((recomputed_nav - ledger.equity).abs().max()) < 1e-7
            assert float(ledger.accounting_error.abs().max()) < 1e-6
            assert (ledger.cash >= -1e-7).all() and (ledger.shares >= 0).all() and (ledger.shares % 100 == 0).all()
            assert ledger.date.iloc[0] == pd.Timestamp("2020-01-02")
            assert ledger.date.iloc[-1] == pd.Timestamp("2026-08-14")
            assert ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
            account_count += 1
        for file in folder.rglob("training_receipts.csv"):
            training = pd.read_csv(file)
            assert (pd.to_datetime(training.last_label_exit) <= pd.to_datetime(training.fit_origin)).all()
            assert (training.train_rows >= 300).all()
            training_count += len(training)
    alignment = pd.read_parquet(root / "reports/research/510300_overnight_global_information_v1/source_clock_alignment.parquet")
    aligned = alignment.loc[alignment.available_at.notna()]
    assert (aligned.available_at <= aligned.decision_time).all()
    for file in [root / "config" / f"{stage}_manifest.json" for stage in STAGES]:
        manifest = json.loads(file.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            target = root / item["path"]
            assert target.stat().st_size == item["bytes"] and sha(target) == item["sha256"], item["path"]
    return {"status": "PASS_SAVED_ACCOUNTS_METRICS_CLOCKS_AND_FROZEN_FILES",
            "evaluation_account_count": account_count, "quarterly_training_receipt_count": training_count,
            "maximum_metric_recomputation_error": max_error, "aligned_source_rows": len(aligned),
            "scope": "结构、时间关系与保存净值的数值复算；未做安全、隐私或外部评审审计"}


def friendly(stage: str, model: str) -> str:
    if stage == STAGES[0]:
        from research.adaptive_allocation_v1 import display_name
        return display_name(model)
    if stage == STAGES[1]:
        from research.return_classification_v1 import label
        return label(model)
    from research.overnight_global_information_v1 import label
    return label(model)


def build() -> None:
    assert not DELIVERY.exists() and not ZIP.exists(), "交付目录已存在，禁止覆盖"
    report = verify_saved(ROOT)
    DELIVERY.mkdir(parents=True)
    results = [json.loads((ROOT / "reports/research" / stage / "result.json").read_text(encoding="utf-8")) for stage in STAGES]
    titles = ["条件切换与滚动回归", "涨跌分类与幅度加权", "免费全球盘后信息"]
    rows = []
    lines = ["# 510300 夏普率 1.2 持续研究：前三轮实际结果", "",
             "**当前还没有实现成本后夏普率 1.2。** 本次已经实际完成三轮研究，共 70 个候选配置。三轮分别保留买入持有基准、基础及压力费用，共计算 146 个完整评价账户；第一轮另保存 50 个连续影子账户供当时的策略选择使用。部分对照配置在不同轮次重复，70 是登记配置数，不是 70 个独立研究机制。", "",
             "统一评价区间为 2020 年 1 月 2 日至 2026 年 8 月 14 日开盘，共 1,604 个交易日。初始 20 万元，只持有 510300 和现金。分红按真实登记、除息和到账处理；等待日、整手、T+1、佣金、最低费用、滑点和退出费用全部纳入。现金参考利率为零。", "",
             "## 每轮主方案的结果", "", "| 研究轮次 | 主方案基础夏普 | 压力夏普 | 基础年化收益 | 基础最大回撤 | 本轮事后最高夏普 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage, title, result in zip(STAGES, titles, results):
        base = next(item for item in result["primary"] if item["cost"] == "BASE")
        stress = next(item for item in result["primary"] if item["cost"] == "STRESS")
        best = result["post_selected_best_base"]
        lines.append(f"| {title} | {base['net_sharpe']:.4f} | {stress['net_sharpe']:.4f} | {base['annualized_return']:.2%} | {base['max_drawdown']:.2%} | {best['net_sharpe']:.4f} |")
        metrics = pd.read_csv(ROOT / "reports/research" / stage / "metrics.csv")
        rows.extend([{**row, "round": title, "strategy_cn": friendly(stage, row["model"])} for row in metrics.to_dict("records")])
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(DELIVERY / "三轮全部候选指标.csv", index=False, encoding="utf-8-sig")
    best = all_metrics.loc[(all_metrics.cost == "BASE") & (all_metrics.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    benchmark = all_metrics.loc[(all_metrics.cost == "BASE") & (all_metrics.model == "BUY_HOLD")].iloc[0]
    lines.extend(["", f"同一口径下买入持有的夏普为 {benchmark.net_sharpe:.4f}，年化收益 {benchmark.annualized_return:.2%}，最大回撤 {benchmark.max_drawdown:.2%}。三轮事后最高读数来自“{best.strategy_cn}”：夏普 {best.net_sharpe:.4f}、年化收益 {best.annualized_return:.2%}。这仍不是达到 1.2 的策略，且事后挑选还有重复试验偏差。", "",
                  "## 这三轮说明了什么", "",
                  "第一，过去一两年表现好的策略，未必适合接下来的市场。第一轮直接按过去表现挑策略，主方案反而亏损，说明切换规则本身也需要可预测的依据。", "",
                  "第二，复杂模型与增加因子都要看实际账户结果。岭回归、浅层梯度提升、极随机树、涨跌分类、幅度加权、概率共识以及海外信息都有完整结果，不能只凭模型名称判断优劣。", "",
                  "第三，盘后信息有部分历史增量，但频繁切换付出的费用很高。第三轮主方案在基础费用下略好于持有基准，费用加倍后明显减弱；这还不足以证明稳定超额。", "",
                  "第四，扩大仓位或杠杆本身不能凭空提高风险调整后的表现。当前差距需要新的有效预测信息或不同的可执行研究机制，不能靠改变夏普计算口径解决。", "",
                  "## 当前研究结论与下一轮条件", "",
                  "前三轮主方案均未达标，原协议、参数和失败结果全部保留。下一轮应先解释新增信息或决策机制为何可能改变结果，再固定有限候选、时间、费用与账户规则；继续用当前完整区间评价，不把历史最好片段单独拿出来。", "",
                  "用户确认的夏普下限为 1.2；年化超额和最大回撤没有新的明确数值要求，本报告不替用户补充。是否允许交易研究扩展到其他 ETF 的问题尚未得到回答，当前继续按 510300 与现金实施。", "",
                  "日线历史已经被多次研究，因此这三轮是时间递进的历史重放，不是全新独立样本。第三轮海外资料另有免费接口历史首次交付时刻未证实的限制。任何历史读数达到 1.2 的候选仍需要明确标明其证据级别，不能直接称为未来稳定可实现。", "",
                  "## 核对与文件", "",
                  f"已复算 {report['evaluation_account_count']} 个评价账户的夏普、复合年化、累计收益和最大回撤；已核对 {report['quarterly_training_receipt_count']} 条季度拟合记录的标签成熟时间、跨市场信息时钟和冻结文件。最大指标复算差为 {report['maximum_metric_recomputation_error']:.3g}。新旧账户引擎的买入持有逐日数字完全一致。", "",
                  "本轮新增的 12 项针对性测试在真实研究前分别通过。仅做结构与数值核对，不把这些检查称为安全审计，也没有将资料上传或声称已经由外部 GPT 审阅。", "",
                  "下面附上三轮全部因子、算法参数、中文交易规则和完整结果。"])
    for protocol, stage in zip(PROTOCOLS, STAGES):
        lines.extend(["", "---", "", (ROOT / "docs" / protocol).read_text(encoding="utf-8")])
        for md in sorted((ROOT / "reports/research" / stage).glob("*实际结果.md")):
            lines.extend(["", md.read_text(encoding="utf-8")])
    report_name = "510300夏普1.2研究_三轮结果与完整中文规则.md"
    (DELIVERY / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# 从这里开始

用户目标：成本后夏普率达到 1.2，并持续推进研究。当前三轮均未达标。

先读 `{report_name}`，再读 `三轮全部候选指标.csv`。完整的协议、代码、固定输入、逐日账户、训练记录和模型在保留原相对路径的文件夹中。净值图是辅助阅读材料，不代替逐日记录。

所有 70 个登记配置和费用情景均保留；一些是重复对照，不能把 70 当作独立试验数。没有隐瞒失败候选，也没有把数据源核对或模型训练完成写成达到投资目标。

核对范围为结构、文件清单、哈希、CRC、保存账户与指标复算，未做安全审计。此包未上传，未取得外部 GPT 审阅结论。真实持仓与交易授权没有改变。

需要复算时，在具有所记录 Python 依赖的环境中运行 `scripts/build_sharpe_1_2_three_round_review_20260906.py --verify-only`。该模式仅检查包内保存结果，不下载资料，不重新搜索模型，不覆盖研究文件。
"""
    (DELIVERY / "00_先读说明.md").write_text(readme, encoding="utf-8")
    prompt = """# 可直接交给 GPT 的审阅请求

请以反方研究员的立场审阅本包。目标是成本后夏普率 1.2，当前三轮实际研究均未达到；交易研究暂限 510300 与现金，来源预算为零。

先阅读总说明与三轮中文协议，再核对训练时间、跨时区信息可用时间、标签成熟、完整账户成本、分红、T+1、下一开盘成交、期末退出和所有失败候选。请区分预先指定主方案与事后最好的模型；这些历史已被多次研究，不能声称是新独立样本。

请具体回答：失败主要来自预测信息不足、策略切换追逐旧赢家、仓位映射、费用、还是其他可定位错误？指出有证据的错误与推测，给出文件路径和可复现核对步骤。不要为了达到 1.2 而建议挑区间、隐瞒试验、删现金日或漏算费用。

随后提出下一轮一个有经济解释的新研究机制，说明与本包已失败方案有什么实质区别。给出全部中文因子定义、信息时钟、训练/验证次序、有限候选、完整账户口径、验收条件和停止条件，并排序下一步工作。新方法建议属于待执行，不得被描述成已有绩效。若需要额外资产、收费来源、券商功能或尚未观察的数据，明确列出依赖。

不要把外部评审通过当成策略已经达到目标，也不要提出真实下单。
"""
    (DELIVERY / "01_GPT审阅提示词.md").write_text(prompt, encoding="utf-8")
    paths = {Path(__file__), ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
             ROOT / "config/510300_research_authority_v5.json", ROOT / "config/510300_research_authority_v6.json"}
    for stage in STAGES:
        manifest_path = ROOT / "config" / f"{stage}_manifest.json"
        paths.add(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths.update(ROOT / item["path"] for item in manifest["files"])
        paths.update(p for p in (ROOT / "reports/research" / stage).rglob("*") if p.is_file())
    for name in ("research/__init__.py", "tests/__init__.py", "scripts/__init__.py"):
        if (ROOT / name).is_file():
            paths.add(ROOT / name)
    coverage = json.loads((ROOT / "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json").read_text(encoding="utf-8"))
    for item in coverage.get("official_source_snapshots", []):
        source = ROOT / item["saved_file"]
        if source.is_file():
            paths.add(source)
    for source in sorted(paths):
        target = DELIVERY / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    draw_equity(results)
    copied_check = verify_saved(DELIVERY)
    assert copied_check == report
    index_rows = []
    for file in sorted(DELIVERY.rglob("*")):
        if file.is_file():
            index_rows.append({"path": file.relative_to(DELIVERY).as_posix(), "bytes": file.stat().st_size, "sha256": sha(file)})
    with (DELIVERY / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index_rows)
    with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(DELIVERY.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(DELIVERY).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert set(names) == {row["path"] for row in index_rows} | {"FILE_INDEX.csv"}
        for row in index_rows:
            content = archive.read(row["path"])
            assert len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
    receipt = {"zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "members": len(names),
               "crc": "PASS", "index": "PASS", "file_hashes": "PASS", "saved_result_recomputation": report,
               "external_review_received": False, "security_audit_performed": False}
    receipt_path = ZIP.with_suffix(".delivery.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def draw_equity(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#b95045", "#4e71a8", "#259181"]
    for stage, result, color, name in zip(STAGES, results, colors, ["第一轮主方案", "第二轮主方案", "第三轮主方案"]):
        model = result["primary"][0]["model"]
        ledger = pd.read_parquet(ROOT / "reports/research" / stage / "evaluation/BASE" / f"{model}_ledger.parquet")
        equity = ledger.equity / 200000
        peaks = np.maximum.accumulate(np.r_[1, equity])[1:]
        axes[0].plot(ledger.date, equity, label=name, color=color, linewidth=1.5)
        axes[1].plot(ledger.date, equity / peaks - 1, color=color, linewidth=1.2)
    ledger = pd.read_parquet(ROOT / "reports/research" / STAGES[0] / "evaluation/BASE/BUY_HOLD_ledger.parquet")
    equity = ledger.equity / 200000
    axes[0].plot(ledger.date, equity, label="买入持有", color="#555555", linestyle="--", linewidth=1.5)
    axes[1].plot(ledger.date, equity / np.maximum.accumulate(np.r_[1, equity])[1:] - 1,
                 color="#555555", linestyle="--", linewidth=1.2)
    axes[0].set_title("510300 三轮预先指定主方案：完整账户净值", loc="left", fontsize=15, pad=15)
    axes[0].set_ylabel("净值（初始为 1）")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_ylabel("相对最高点回撤")
    from matplotlib.ticker import PercentFormatter
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.09, 0.025, "2020-01-02 至 2026-08-14 开盘；基础费用；现金收益为零；含分红、等待日与退出费用。\n历史已被多轮研究，图中结果不是未来收益承诺。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(DELIVERY / "三轮主方案_净值与回撤.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成中文审阅包或复算保存结果")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(ROOT), ensure_ascii=False), flush=True)
    else:
        build()
