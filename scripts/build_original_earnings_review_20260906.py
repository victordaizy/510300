"""形成第八轮中文说明与可离线复算保存结果的审阅包。"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "reports/research/510300_original_earnings_breadth_v1"
FUND = ROOT / "reports/research/510300_fundamental_and_fund_flow_rebuild_v1"
COMPLETION = ROOT / "reports/research/510300_original_earnings_source_completion_v1"
FACTOR = ROOT / "reports/research/510300_factor_definition_review_20260906"
DELIVERY = ROOT / "deliverables/510300第八轮_原始财报盈利广度_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_第八轮原始财报_GPT审阅_20260906.zip"


def physical(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    outside = Path(r"E:\ResearchData\New project 8") / rel
    return outside if rel.parts[0] == "data" and outside.exists() else path


def label(key):
    from research.original_earnings_breadth_v1 import NAMES
    if key in NAMES:
        return NAMES[key]
    group, kind, horizon = key.split("_")
    return {"PRICE": "价格信息", "EARNINGS": "原始财报信息", "FULL": "价格与原始财报"}[group] + "／" + {"RIDGE": "岭回归", "ET": "极随机树"}[kind] + "／" + {"H20": "二十日", "H60": "六十日"}[horizon]


def number(value, digits=4):
    return "无波动，夏普不定义" if pd.isna(value) else f"{value:.{digits}f}"


def verify_saved():
    from research.adaptive_allocation_v1 import summarize
    config = json.loads((ROOT / "config/510300_original_earnings_breadth_v1.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(REPORT / "metrics.csv")
    maximum = 0.0
    for item in frame.to_dict("records"):
        ledger = pd.read_parquet(REPORT / "evaluation" / item["cost"] / (item["model"] + "_ledger.parquet"))
        actual = summarize(ledger, config)
        for key, value in actual.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
                err = abs(float(item[key]) - value)
                assert err < 1e-7, (item["model"], key, err)
                maximum = max(maximum, err)
        assert len(ledger) == 1604 and ledger.date.min() == pd.Timestamp("2020-01-02") and ledger.date.max() == pd.Timestamp("2026-08-14")
        assert ledger.accounting_error.abs().max() < 1e-6
        if item["model"] == "BUY_HOLD":
            old = pd.read_parquet(ROOT / "reports/research/510300_adaptive_allocation_v1/evaluation" / item["cost"] / "BUY_HOLD_ledger.parquet")
            for col in ["equity", "shares", "cash", "net_return", "commission", "slippage_cost"]:
                np.testing.assert_array_equal(ledger[col], old[col])
    fits = pd.read_json(REPORT / "training_receipts.json")
    trained = fits.loc[fits.status.eq("FITTED")]
    assert len(fits) == 336 and len(trained) == 138
    assert (pd.to_datetime(trained.last_label_exit) <= pd.to_datetime(trained.fit_origin)).all()
    assert trained.train_rows.ge(300).all()
    assert fits.loc[~fits.status.eq("FITTED"), "train_rows"].lt(300).all()
    for _, group in fits.groupby(["horizon", "fit_origin"]):
        assert group.train_rows.nunique() == 1 and group.status.nunique() == 1
        if group.status.iloc[0] == "FITTED":
            assert group.last_label_exit.nunique() == 1 and group.last_train_origin.nunique() == 1
    data = pd.read_parquet(REPORT / "features.parquet")
    predictions = pd.read_parquet(REPORT / "predictions.parquet")
    assert predictions.loc[~data.feature_valid, predictions.columns != "date"].isna().all().all()
    return {"status": "PASS_SAVED_ACCOUNT_METRICS_LABEL_MATURITY_AND_NO_VIEW", "accounts": len(frame),
            "scheduled_fits": len(fits), "completed_fits": len(trained), "no_view_fit_origins": len(fits) - len(trained),
            "maximum_metric_error": maximum, "benchmark_daily_parity": "EXACT",
            "retrained_models": False, "security_audit_performed": False}


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for key, name, color in [("F1_PRIMARY_FULL_H60", "价格与财报主方案", "#167c80"), ("F2_EARNINGS_H60", "仅财报对照", "#a56633"), ("BUY_HOLD", "买入持有", "#73767d")]:
        d = pd.read_parquet(REPORT / "evaluation/BASE" / (key + "_ledger.parquet"))
        wealth = d.equity.to_numpy() / 200000
        high = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        axes[0].plot(d.date, wealth, color=color, label=name, linewidth=1.6)
        axes[1].plot(d.date, 100 * (wealth / high - 1), color=color, linewidth=1.25)
    axes[0].set_title("第八轮：原始财报盈利改善广度", loc="left", pad=43, fontsize=16)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    axes[0].set_ylabel("完整账户净值（初始为1）")
    axes[1].set_ylabel("回撤（%）")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.text(.08, .02, "2020-01-02 至 2026-08-14 开盘；基础费用，分红、等待日与期末退出均计入。\n原始财报覆盖不足导致较长无判断时期；主方案2024年起有预测，2025年12月才首次实际持有。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(DELIVERY / "第八轮_完整账户净值与回撤.png", dpi=150)
    plt.close(fig)


def research_imports(paths):
    found = set(paths)
    todo = [p for p in found if p.suffix == ".py"]
    while todo:
        path = todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] if node.module != "research" else ["research." + x.name for x in node.names]
            for name in names:
                if name.startswith("research."):
                    child = ROOT / (name.replace(".", "/") + ".py")
                    if child.exists() and child not in found:
                        found.add(child)
                        todo.append(child)
    init = ROOT / "research/__init__.py"
    if init.exists():
        found.add(init)
    return found


def build():
    assert not DELIVERY.exists() and not ZIP.exists(), "交付已存在，不覆盖"
    checked = verify_saved()
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    source = json.loads((REPORT / "source_receipt.json").read_text(encoding="utf-8"))
    completion = json.loads((COMPLETION / "batch_01_result.json").read_text(encoding="utf-8"))
    queue = json.loads((COMPLETION / "batch_01_protocol.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(REPORT / "metrics.csv")
    eras = pd.read_csv(REPORT / "era_metrics.csv")
    base, stress = [next(x for x in result["primary"] if x["cost"] == c) for c in ("BASE", "STRESS")]
    DELIVERY.mkdir(parents=True)
    lines = ["# 第八轮：原始财报盈利改善广度", "",
             f"**主方案成本后夏普为{base['net_sharpe']:.4f}，尚未达到1.2。** 完整账户年化收益{base['annualized_return']:.2%}，最大回撤{-base['max_drawdown']:.2%}；提高佣金和滑点后，夏普为{stress['net_sharpe']:.4f}。", "",
             "本轮实际完成16个候选、34条完整账户。主方案事先固定为：把510300价格信息与当时已公开的成分股原始财报信息合并，预测未来60个交易日总回报，再结合波动和换仓成本决定持仓。", "",
             f"本轮事后最高夏普为{result['post_selected_best_base']['net_sharpe']:.4f}，来自“{label(result['post_selected_best_base']['model'])}”。所有结果均保留，不能把事后最佳替换成预设主方案。", "",
             "## 给老板看的解释", "",
             "股票价格可以从每股盈利和市场愿意支付的市盈率两端理解；投资者的持有总回报还包含现金分红。回购会影响股数、每股盈利和股东权益，必须防止重复计入收益。公募申购则是可能影响需求的另一层信息，需要区分基金份额、真实现金流和净值上涨。", "",
             "本轮先做盈利这一端：如果越来越多沪深300成分公司的利润、扣非利润、收入和经营现金流都好转，510300未来回报是否更容易改善？原始财报组记录这些改善的普遍程度，并与只看价格的方案比较。", "",
             "目前仍有两个明显限制。第一，原始档案主要来自此前预先排除金融公司的研究，缺失集中于银行、保险、证券公司以及部分新调入成分；它不是完整指数盈利。第二，足够可靠的财报覆盖来得较晚，主方案直到2024年1月才形成预测，2025年12月才第一次实际持有。小回撤主要伴随着很少的持仓，不能解释为已经找到了高效的赚钱能力。", "",
             "## 本轮数据到底用了什么", "",
             f"读取11891个原始公告对应的事实记录，选取{source['selected_fact_rows']:,}项财务事实；其中{source['amounts_independently_recomputed']:,}项从原文金额和单位重新换算核对。{source['blank_or_non_numeric_rows_excluded']}项空白或不可按直接数字解析的记录不作为有效金额，另有3项不能对应主文件或已确认配套文件的事实保留为缺失。原始档案未通过旧研究覆盖门槛的结论保持原状。", "",
             "五项底层财务量是归母净利润、扣非归母净利润、营业收入、经营活动现金流量净额和期末总资产。它们与上一年相同报告期原始金额比较；不用今天供应商修订值回填，也不用现在的300家公司替换过去成分。原始PDF地址、页码、单位、片段与哈希一并交付。", "",
             "只有日期的公告，保守地从公布日之后第一个A股交易日开盘起使用。先查该公司当时最新报告，再检查事实是否齐全，不能在新报告缺数时退回旧报告凑覆盖。每日至少240家公司有效才允许预测，缺失判断的日期仍留在完整账户。", "",
             "历史覆盖首次达到240家的日期是2022年5月5日。336次预定季度拟合中，138次满足成熟训练样本要求并实际拟合，198次因资料或已到期训练样本不足保持无判断。16个候选始终使用同一批有效日期。", "",
             "## 全部候选表现", "",
             "| 策略 | 基础夏普 | 压力夏普 | 基础年化收益 | 相对买入持有年化差 | 基础最大回撤 | 成交笔数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    from research.original_earnings_breadth_v1 import ENSEMBLES
    order = [*ENSEMBLES, "BUY_HOLD"] + [x for x in metrics.model.unique() if x not in ENSEMBLES and x != "BUY_HOLD"]
    for key in order:
        a, b = [metrics.loc[(metrics.model == key) & (metrics.cost == c)].iloc[0] for c in ("BASE", "STRESS")]
        lines.append(f"| {label(key)} | {number(a.net_sharpe)} | {number(b.net_sharpe)} | {a.annualized_return:.2%} | {a.annualized_return_excess_vs_buy_hold:.2%} | {-a.max_drawdown:.2%} | {a.trade_count} |")
    lines += ["", "本表全部采用2020年1月2日至2026年8月14日完整账户，不能与剔除等待日后的收益直接比较。两档费用会影响考虑换仓成本的持仓选择，因此压力路径可能与基础路径不同。", "",
              "## 主方案分期与不确定性", "", "| 阶段 | 基础夏普 | 年化收益 | 最大回撤 |", "|---|---:|---:|---:|"]
    for r in eras.loc[(eras.model == "F1_PRIMARY_FULL_H60") & (eras.cost == "BASE")].itertuples():
        lines.append(f"| {r.era} | {number(r.net_sharpe)} | {r.annualized_return:.2%} | {-r.max_drawdown:.2%} |")
    u = result["uncertainty"]["BASE"]
    lines += ["", f"主方案夏普的95%区间为{u['primary_sharpe_95_interval'][0]:.4f}至{u['primary_sharpe_95_interval'][1]:.4f}。相对同样资料有效日期的价格等权对照，年化日均收益增量区间为{u['increment_vs_price_95_interval'][0]:.2%}至{u['increment_vs_price_95_interval'][1]:.2%}，跨过零。没有确认稳定增益。", "",
              "这些区间没有消除历次查看历史和尝试多个候选的选择影响；不是新的独立样本验证。", "",
              "## 盈利、估值、股东回报和公募申购的后续落实", "",
              f"已生成{queue['missing_original_documents']:,}份原始报告的具体缺口清单，涉及{queue['affected_symbols']}家公司。首批按资料缺失影响选择12家公司、24份报告，已成功归档{completion['archived']}份，并提取前十二页文字供逐项核实。包括银行、券商和保险公司。新资料尚未作为合格因子，也未回填本轮账户。", "",
              "金融公司将优先使用适合自身报表的利润、每股盈利、净资产和权益回报口径，与实体企业的现金流质量分别处理。随后恢复股份变化和分红回购的历史公告，才能正确解释每股盈利和股东回报。估值资料也要保留版本和可用时间，避免用价格除以市盈率的恒等变换冒充新信息。", "",
              "公募方面已结构化77份协会月报。对PDF元数据继续核对后，发现其中44份早期报告的现存文件标记为2023年11月21日生成，尚不能证明它们的当前表内数值在原历史日期已经公开。保留全部记录、5处相邻月报份额修订差异和2025年11月分类调整；继续恢复原始发布日期，不在未证明时钟的旧数据上假装完成申购因子验证。", "",
              "## 本轮所有因子与持仓规则", "", (ROOT / "docs/510300_ORIGINAL_EARNINGS_BREADTH_V1_PROTOCOL.md").read_text(encoding="utf-8"), "",
              "## 三十四项价格对照因子的中文定义", "",
              "这些既有价格因子在上一轮已经从价格和分红原始资料独立复算。本轮复用同一文件，不重复改参数。下表每行是一项输入，程序字段不是策略规则。", "",
              "| 序号 | 中文计算规则 | 单位或经济含义 | 解释边界 |", "|---:|---|---|---|"]
    price = pd.read_csv(FACTOR / "95项因子逐项中文核对.csv").query('因子组 == "价格与财富"')
    assert len(price) == 34
    for i, row in enumerate(price.to_dict("records"), 1):
        values = [str(row[c]).replace("|", "／") for c in ("中文定义", "单位或含义", "问题或使用边界")]
        lines.append(f"| {i} | {' | '.join(values)} |")
    lines += ["", "## 交付验证与资料范围", "",
              "五项针对性测试通过，覆盖原文金额及单位、亏损转盈利的变化方向、下一个开盘日可用时钟、最新报告缺失不得退回旧报告、历史成分变化与未来追加不改变过去特征。保存的34条账户指标复算通过，买入持有逐日相同，138次有效训练的标签全部已到期。", "",
              "本包可以离线复算已保存的账户指标并核对冻结输入。财务事实源档保留原始文件地址、页码、片段和哈希；没有重新下载解析全部11891份PDF，不能将本次金额换算核对说成重新逐页查验全部原始报告。新收集的24份PDF和77份公募月报在包内。", "",
              "只做必要的结构与数值检查，未做安全审计，未上传，也未收到外部审阅。研究和模拟不构成真实持仓或交易指令。目标未达到，资料修正与新机制研究继续。"]
    (DELIVERY / "第八轮结果与全部中文因子规则.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DELIVERY / "00_先读说明.md").write_text("# 阅读顺序\n\n先读《第八轮结果与全部中文因子规则》，再看净值图、全部指标及逐日账户。根目录保存原相对路径，便于检查代码、配置、原始财务事实与来源。\n\n新下载的金融公司原始财报属于后续来源补齐，未进入第八轮。公募月报也仍处于原始版本时钟恢复阶段。\n\n运行本包内打包脚本的 --verify-only 可离线复算保存账户指标，不训练、不下载。FILE_INDEX.csv 覆盖包内全部其他文件。\n\n完整原始财报PDF库未保留，本包给出既有逐份提取事实、页面片段、网址和哈希；不能把离线复算当作重新验证所有原始PDF。\n", encoding="utf-8")
    (DELIVERY / "01_GPT审阅提示词.md").write_text("# 请审阅第八轮，并提出下一轮可执行研究\n\n目标是仅使用510300与现金、免费数据的完整账户成本后夏普1.2。第八轮16个候选全部未达标。请先检查原始报告主公告与同日配套文件的关联、金额单位、上一年同期原始版本、日期只有天精度时的可用时钟，以及最新报告缺失不能退回旧报告。\n\n重点辨别：源档原先排除了金融公司，改善广度不能代表全指数盈利。240家公司门槛和至少300条成熟训练样本带来了长等待期；请检验为何其余年份全现金、主方案2025年12月才买入，不能把低回撤当高效率。检查全部候选、无判断日、同数据覆盖价格对照、真实分红和费用是否完整。\n\n结合包内具体原始财报缺口、首批24份金融公司报告和公募月报版本记录，提出下一轮一个能实际执行的新研究：分别定义实体企业与金融公司的盈利，处理每股盈利、市盈率、分红回购、股本变化及基金申赎。明确哪些是恒等关系、哪些是额外信息。不能使用未证明原始时钟的公募数据回填过去。\n\n请用中文列出完整因子、来源、可用日期、有限候选、验证与失败后的下一步。保留旧失败，不以改窗口、挑最好时期、倒转符号、删成本或不断调阈值让指标达标。建议不构成真实交易授权。\n", encoding="utf-8")
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    draw()
    manifestpath = ROOT / "config/510300_original_earnings_breadth_v1_manifest.json"
    expected = {r["path"]: r for r in json.loads(manifestpath.read_text(encoding="utf-8"))["files"]}
    paths = {ROOT / rel for rel in expected} | {manifestpath, Path(__file__), ROOT / "scripts/collect_original_earnings_gaps_20260906.py", ROOT / "scripts/record_amac_pdf_version_evidence_20260906.py"}
    for folder in (REPORT, FUND, COMPLETION):
        paths.update(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith("_preview.png"))
    paths.update({FACTOR / "95项因子逐项中文核对.csv", FACTOR / "price_recomputation.json", ROOT / "config/csi300_pit_fundamental_underreaction_enhancement_v1.yaml",
                  ROOT / "reports/research/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1.json"})
    parent = ROOT / "config/510300_adaptive_allocation_v1_manifest.json"
    paths.add(parent)
    paths.update(ROOT / r["path"] for r in json.loads(parent.read_text(encoding="utf-8"))["files"])
    for cost in ("BASE", "STRESS"):
        paths.add(ROOT / "reports/research/510300_adaptive_allocation_v1/evaluation" / cost / "BUY_HOLD_ledger.parquet")
    source_root = Path(r"E:\ResearchData\New project 8")
    for child in ("data/raw/510300_fundamental_and_fund_flow_rebuild_v1/amac", "data/raw/510300_original_earnings_source_completion_v1"):
        paths.update(ROOT / p.relative_to(source_root) for p in (source_root / child).rglob("*") if p.is_file())
    paths = research_imports(paths)
    index = []
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            rel = path.relative_to(ROOT).as_posix()
            content = physical(path).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if rel in expected:
                assert digest == expected[rel]["sha256"], "冻结输入变化：" + rel
            target = DELIVERY / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            archive.writestr(rel, content)
            index.append({"path": rel, "bytes": len(content), "sha256": digest})
        for path in sorted(p for p in DELIVERY.iterdir() if p.is_file()):
            content = path.read_bytes()
            archive.writestr(path.name, content)
            index.append({"path": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index)
        content = stream.getvalue().encode("utf-8-sig")
        (DELIVERY / "FILE_INDEX.csv").write_bytes(content)
        archive.writestr("FILE_INDEX.csv", content)
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        assert archive.testzip() is None
        assert len(names) == len(set(names)) and set(names) == {r["path"] for r in index} | {"FILE_INDEX.csv"}
        for row in index:
            content = archive.read(row["path"])
            assert len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
    receipt = {"zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
               "members": len(names), "crc": "PASS", "index": "PASS", "hashes": "PASS", "recomputation": checked,
               "security_audit_performed": False, "external_review_received": False}
    ZIP.with_suffix(".delivery.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第八轮中文说明、账户复算与完整交付")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved(), ensure_ascii=False), flush=True)
    else:
        build()
