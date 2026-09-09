"""第四轮央行操作量研究的保存结果复算与自包含交付。"""
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
STAGE = "510300_policy_liquidity_quantity_v1"
REPORT = ROOT / "reports/research" / STAGE
DELIVERY = ROOT / "deliverables/510300夏普1.2持续研究_第四轮央行操作量_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_第四轮央行操作量_GPT审阅_20260906.zip"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_saved(root: Path) -> dict:
    print(f"开始核对保存账户：{root}", flush=True)
    folder = root / "reports/research" / STAGE
    metrics = pd.read_csv(folder / "metrics.csv")
    maximum, account_count = 0.0, 0
    for row in metrics.itertuples():
        ledger = pd.read_parquet(folder / "evaluation" / row.cost / f"{row.model}_ledger.parquet")
        r = ledger.net_return.to_numpy(float)
        assert len(r) == 1604 and np.isfinite(r).all() and (r > -1).all()
        wealth = np.r_[1, np.cumprod(1 + r)]
        values = {"net_sharpe": r.mean() / r.std(ddof=1) * np.sqrt(242) if r.std(ddof=1) > 1e-15 else np.nan,
                  "annualized_return": wealth[-1] ** (242 / len(r)) - 1,
                  "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
                  "cumulative_return": wealth[-1] - 1}
        for key, value in values.items():
            recorded = getattr(row, key)
            if pd.isna(value):
                assert pd.isna(recorded)
            else:
                error = abs(value - recorded)
                assert error < 1e-10
                maximum = max(maximum, error)
        assert float((ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable - ledger.equity).abs().max()) < 1e-7
        assert ledger.date.iloc[0] == pd.Timestamp("2020-01-02") and ledger.date.iloc[-1] == pd.Timestamp("2026-08-14")
        assert (ledger.cash >= -1e-7).all() and (ledger.shares >= 0).all() and (ledger.shares % 100 == 0).all()
        assert float(ledger.accounting_error.abs().max()) < 1e-6
        decisions = pd.read_parquet(folder / "evaluation" / row.cost / f"{row.model}_decisions.parquet")
        invalid = decisions.view == "NO_VIEW"
        assert decisions.loc[invalid, "reference_weight"].isna().all()
        assert (decisions.loc[invalid, "requested_quantity"] == 0).all()
        if row.model == "BUY_HOLD":
            old = pd.read_parquet(root / "reports/research/510300_adaptive_allocation_v1/evaluation" / row.cost / "BUY_HOLD_ledger.parquet")
            for column in ("equity", "net_return", "cash", "shares", "dividend_receivable", "dividend_paid", "commission", "slippage_cost"):
                assert np.array_equal(ledger[column], old[column])
        account_count += 1
    training = pd.read_csv(folder / "training_receipts.csv")
    print(f"已核对 {account_count} 个账户，开始核对训练与来源", flush=True)
    assert (pd.to_datetime(training.last_label_exit) <= pd.to_datetime(training.fit_origin)).all()
    assert (training.train_rows >= 300).all()
    for _, part in training.groupby(["fit_origin", "horizon"]):
        assert part.training_origin_sha256.nunique() == 1
    data = pd.read_parquet(folder / "features.parquet")
    for time in ("dr_available_at", "quantity_published_at", "rate_published_at"):
        valid = data[time].notna()
        assert (data.loc[valid, time] <= data.loc[valid, "origin_time"]).all()
    receipt = json.loads((folder / "quantity_source_admission.json").read_text(encoding="utf-8"))
    assert receipt["explicit_zero_count"] == 6 and receipt["absent_7d_row_count"] == 663
    manifest = json.loads((root / "config/510300_policy_liquidity_quantity_v1_manifest.json").read_text(encoding="utf-8"))
    for number, item in enumerate(manifest["files"], 1):
        path = root / item["path"]
        (REPORT / "DELIVERY_IO_CURRENT.txt").write_text(str(path), encoding="utf-8")
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
        if number % 500 == 0:
            print(f"已核对冻结文件 {number}/{len(manifest['files'])}", flush=True)
    return {"status": "PASS_SAVED_ACCOUNT_AND_COMMON_TRAINING_RECOMPUTATION",
            "accounts": account_count, "training_records": len(training), "maximum_metric_error": maximum,
            "benchmark_daily_parity": "EXACT", "same_training_origins_in_three_groups": True,
            "quantity_missing_not_zero": True, "scope": "必要结构、时间及数值检查；未做安全审计"}


def build(resume: bool = False) -> None:
    assert (not DELIVERY.exists() or resume) and not ZIP.exists(), "已交付路径禁止覆盖"
    verified = verify_saved(ROOT)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config/510300_policy_liquidity_quantity_v1.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(REPORT / "metrics.csv")
    from research.policy_liquidity_quantity_v1 import label
    a = metrics.loc[(metrics.cost == "BASE") & (metrics.model == "Q1_PRIMARY_FULL_H20")].iloc[0]
    b = metrics.loc[(metrics.cost == "STRESS") & (metrics.model == "Q1_PRIMARY_FULL_H20")].iloc[0]
    funding = metrics.loc[(metrics.cost == "BASE") & (metrics.model == "Q2_FUNDING_H20")].iloc[0]
    best = result["post_selected_best_base"]
    DELIVERY.mkdir(parents=True, exist_ok=resume)
    point_target = bool(a.meets_point_target)
    overview = f"""# 510300 夏普率 1.2 持续研究：第四轮结果

本轮研究**央行七天逆回购已披露操作量与资金压力的交互**，不同于前三轮的纯价格与全球价格信息。已经实际运行 16 个候选、两种费用与买入持有基准，共 34 个完整评价账户。

预先指定主方案基础夏普为 **{a.net_sharpe:.4f}**，压力夏普为 **{b.net_sharpe:.4f}**；基础年化收益 **{a.annualized_return:.2%}**，年化超额 **{a.annualized_return_excess_vs_buy_hold:.2%}**，最大回撤 **{a.max_drawdown:.2%}**。基础夏普是否达到 1.2：**{'达到历史点估计目标，仍需独立验证' if point_target else '未达到'}**。

不含操作量的资金对照组共识夏普为 **{funding.net_sharpe:.4f}**。操作量组相对该对照组的年化日均增量，其 95% 区块重抽样区间为 **{result['uncertainty']['BASE']['annualized_arithmetic_increment_vs_funding_95_interval']}**。这项配对对照用于检验操作量有没有额外信息，不能仅凭一个候选的名次判断。

本轮事后最高夏普来自“{label(best['model'])}”，基础夏普 **{best['net_sharpe']:.4f}**。事后最好者不自动替代主方案，也不是新的独立验证结果。

评价仍为 2020 年 1 月 2 日至 2026 年 8 月 14 日开盘，共 1,604 个交易日、20 万元起始资金、只持有 510300 和现金。完整计入分红、等待日、无有效判断日、佣金、最低费用、滑点和退出费用。

本轮新增来源核对重解析了 2,773 条归档央行公告，保留 6 条明确零操作量，另有 663 条没有七天操作行的公告不填零。研究的是七天品种的已披露规模，不是央行总投放或净投放。资金利率只使用 DR007，不使用名称相近代理。历史资料的实时首次交付能力仍未得到证明。

共核对 {verified['accounts']} 个保存账户及 {verified['training_records']} 条季度训练记录，三组相同期限与时点的训练起点完全相同，资金与公告可用时间未越过决策时点。买入持有逐日记录与前三轮完全一致。5 项针对性测试已在真实收益运行前通过。仅进行必要的结构、时间和数值核对，未做安全审计，未上传，未声称外部评审已完成。

前三轮的 70 个登记配置与本轮 16 个配置合计 86 个，包含重复对照，不能按 86 个独立研究机制理解。旧估值、宏观压力、价格模型的失败记录均保留。持续研究尚不能被当成目标已经实现，下一轮须提出可检验的实质新机制。

以下附上完整中文规则与全部结果。
"""
    protocol = (ROOT / "docs/510300_POLICY_LIQUIDITY_QUANTITY_V1_PROTOCOL.md").read_text(encoding="utf-8")
    price_protocol = (ROOT / "docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md").read_text(encoding="utf-8")
    price_factors = price_protocol.split("## 所有因子如何计算", 1)[1].split("## 12 个规则策略", 1)[0]
    details = (REPORT / "510300央行操作量与资金压力_实际结果.md").read_text(encoding="utf-8")
    main_name = "第四轮结果与全部中文因子规则.md"
    (DELIVERY / main_name).write_text(overview + "\n---\n\n" + protocol + "\n## 附录：价格对照组的 34 个因子\n" + price_factors + "\n---\n\n" + details, encoding="utf-8")
    (DELIVERY / "00_先读说明.md").write_text(f"# 从这里开始\n\n先读 `{main_name}`，再读 `reports/research/{STAGE}/metrics.csv`。全部代码、协议、原始公告、DR007 响应、价格与分红资料按原相对路径保留。\n\n包内不是整个项目历史镜像，只完整交付本轮及其直接依赖。前三轮总体结果保留原包。所有账户包括失败候选，历史已被多次研究，不能称为独立未见样本。\n\n使用所记录的 Python 依赖，在包根目录运行本脚本的 --verify-only 模式可以复算保存结果；不会重新训练、下载或改写研究。检查范围为结构、时钟、数值与冻结身份，没有进行安全审计或外部 GPT 审阅。\n", encoding="utf-8")
    prompt = """# 交给 GPT 的审阅请求

请审阅本包第四轮研究。目标为 510300 与现金的完整账户成本后夏普 1.2，免费来源，不授权真实交易。先读中文总说明与协议，再核对原始公告、模型、逐日账户和全部候选。

请特别检查：没有七天操作行是否被错误当成零；研究是否误称净投放；DR007 是否为准确的成交利率；收盘后公告与资金利率的次日可用规则是否正确；二十个历史公告均值是否排除了当前公告；三个输入组训练样本是否一致；缺失信息是否被伪造成现金信号；费用、分红、T+1 和退出是否计入；是否把对照组或事后最佳策略偷换为主方案。

请给出证据支持的错误、代码与文件位置和复算步骤。判断操作量有没有额外预测价值，并把点估计、区间、重复搜索及实时可取得性限制分别解释。

提出下一轮一个有经济机制且与本轮、前三轮实质不同的方案，给出中文因子定义、数据时钟、有限候选、训练与完整账户、优先级、验收和停止条件。不得建议挑窗口、改费用、只保留最佳候选来凑到 1.2。新建议标记为未执行，不写成已取得收益。不要要求实盘下单，不把本次外部审阅等同于投资目标完成。
"""
    (DELIVERY / "01_GPT审阅提示词.md").write_text(prompt, encoding="utf-8")
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = ROOT / "config/510300_policy_liquidity_quantity_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {ROOT / item["path"] for item in manifest["files"]}
    paths.update([manifest_path, Path(__file__)])
    paths.update(p for p in REPORT.rglob("*") if p.is_file() and p.name != "DELIVERY_IO_CURRENT.txt")
    for cost in ("BASE", "STRESS"):
        paths.add(ROOT / "reports/research/510300_adaptive_allocation_v1/evaluation" / cost / "BUY_HOLD_ledger.parquet")
    for filename in ("research/__init__.py", "tests/__init__.py", "scripts/__init__.py"):
        if (ROOT / filename).is_file():
            paths.add(ROOT / filename)
    coverage = json.loads((ROOT / config["inputs"]["dividend_coverage"]).read_text(encoding="utf-8"))
    for item in coverage.get("official_source_snapshots", []):
        path = ROOT / item["saved_file"]
        if path.is_file():
            paths.add(path)
    for source in sorted(paths):
        target = DELIVERY / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    draw_figure()
    assert verify_saved(DELIVERY) == verified
    print("包内保存账户与冻结文件复算完成，开始建立索引与压缩", flush=True)
    index = [{"path": p.relative_to(DELIVERY).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(DELIVERY.rglob("*")) if p.is_file()]
    with (DELIVERY / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index)
    with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(DELIVERY.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(DELIVERY).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        assert archive.testzip() is None and len(names) == len(set(names))
        assert set(names) == {row["path"] for row in index} | {"FILE_INDEX.csv"}
        for row in index:
            payload = archive.read(row["path"])
            assert len(payload) == row["bytes"] and hashlib.sha256(payload).hexdigest() == row["sha256"]
    receipt = {"zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP), "members": len(names),
               "crc": "PASS", "index": "PASS", "hashes": "PASS", "recomputation": verified,
               "security_audit_performed": False, "external_review_received": False}
    ZIP.with_suffix(".delivery.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def draw_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, name, color in (("Q1_PRIMARY_FULL_H20", "新增操作量的主方案", "#167d91"),
                              ("Q2_FUNDING_H20", "不含操作量的资金对照", "#c78732"),
                              ("BUY_HOLD", "买入持有", "#60646b")):
        ledger = pd.read_parquet(REPORT / "evaluation/BASE" / f"{key}_ledger.parquet")
        ax.plot(ledger.date, ledger.equity / 200000, label=name, color=color, linewidth=1.6,
                linestyle="--" if key == "BUY_HOLD" else "-")
    ax.set_title("第四轮：央行操作量是否带来额外收益", loc="left", fontsize=15, pad=45)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    ax.set_ylabel("完整账户净值（初始为 1）")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.03, "2020-01-02 至 2026-08-14 开盘；基础费用；含分红、现金等待日和退出费用。\n历史已被多次研究；本图不代表未来可稳定达到夏普 1.2。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(DELIVERY / "第四轮操作量信息对照_净值.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交付第四轮研究或复算包内保存结果")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume-delivery", action="store_true", help="研究已完成，继续尚未形成压缩包的交付")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(ROOT), ensure_ascii=False), flush=True)
    else:
        build(resume=args.resume_delivery)
