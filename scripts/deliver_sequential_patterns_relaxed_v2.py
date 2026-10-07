"""整理V2结果并构建包含原V1快照的单一GPT复核包。"""
from __future__ import annotations
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sequential_patterns_2021_2026_v2"
PARENT = ROOT / "reports/research/510300_sequential_patterns_regime_v1"
ZIP = ROOT / "deliverables/510300_连续形态启停_V2_2021至2026_GPT审阅_20260924.zip"
sys.path.insert(0, str(OUT / "code"))
from sequential_patterns_regime_v1 import digest, now, save_csv, save_json


def write(name, text):
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def pct(x):
    return f"{0.0 if x == 0 else x:.2%}"


def num(x):
    return "未定义" if pd.isna(x) else f"{x:.3f}"


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
    specs = [("baseline", "PATTERN_ONLY", "仅形态 · 40笔", "#176b79"),
             ("baseline", "STATE_ONLY", "固定状态 · 25笔", "#b27425"),
             ("relaxed", "FULL", "V2完整启停 · 0笔", "#b24e52"),
             ("baseline", "BUY_HOLD", "买入持有", "#85909b")]
    for folder, policy, label, color in specs:
        d = pd.read_parquet(OUT / folder / "accounts/STRESS" / policy / "daily.parquet")
        x = pd.to_datetime(d.date)
        axes[0].plot(x, d.equity_cny / 200000, label=label, color=color, lw=1.8)
        axes[1].plot(x, d.drawdown, color=color, lw=1.4)
    axes[0].set_title("2021—2026：降低样本门槛仍未建立有效启停", loc="left", fontsize=17, pad=15)
    axes[0].set_ylabel("20万元完整账户净值")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1].set_ylabel("历史高点回撤")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axhline(-.1, color="#b24e52", ls="--", alpha=.55, lw=1)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["right", "top"]].set_visible(False)
    fig.text(.08, .026, "2021-01-04至2026-09-16，压力成本，包含空仓日。V2仅改变案例数量6→3、同状态4→2。\n"
             "仅形态年化3.66%、夏普0.491、回撤11.19%；完整启停0笔。历史反复使用，独立前向样本0。", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(OUT / "2021至2026账户对比.png", dpi=165)
    fig.savefig(OUT / "2021至2026账户对比.svg")
    plt.close(fig)


def main():
    if ZIP.exists():
        raise RuntimeError("V2最终交付包已存在，不覆盖。")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(OUT / "comparison.csv")
    annual = []
    for folder in ("baseline", "relaxed"):
        for path in (OUT / folder / "accounts").rglob("daily.parquet"):
            d = pd.read_parquet(path)
            trades = pd.read_csv(path.parent / "trades.csv")
            for year in sorted(d.date.str[:4].unique()):
                y = d[d.date.str.startswith(year)]
                tr = trades[trades.entry_date.str.startswith(year)] if len(trades) else trades
                annual.append({"variant": folder, "cost": path.parent.parent.name, "policy": path.parent.name,
                               "year": year, "sessions": len(y), "net_return": float(np.prod(1 + y.daily_return) - 1),
                               "completed_cycles_by_entry_year": len(tr), "partial_year": year == "2026", "minimum_cycle_quota": None})
    save_csv(OUT / "annual.csv", annual)
    t = pd.read_csv(OUT / "relaxed/trigger_decisions.csv")
    save_csv(OUT / "trigger_gate_reasons.csv", t.groupby(["family", "reason"]).size().rename("count").reset_index())
    counts = t.groupby("reason").size().to_dict()
    single = pd.read_csv(OUT / "relaxed/accounts/STRESS/RECENT_ONLY/trades.csv").iloc[0]
    pc = pd.read_csv(OUT / "baseline/accounts/STRESS/PATTERN_ONLY/trades.csv")
    concentration = {"definition": "完整交易周期人民币净利润归因，非固定事件窗口归因。",
                     "total_net_pnl_cny": float(pc.net_pnl.sum()),
                     "largest_trade": pc.loc[pc.net_pnl.idxmax()].to_dict(),
                     "largest_share": float(pc.net_pnl.max() / pc.net_pnl.sum()),
                     "top3_share": float(pc.nlargest(3, "net_pnl").net_pnl.sum() / pc.net_pnl.sum())}
    save_json(OUT / "profit_concentration.json", concentration)
    table = ["|方案|基础净年化|基础夏普|压力净年化|压力夏普|压力最大回撤|完整周期|", "|---|---:|---:|---:|---:|---:|---:|"]
    specs = [("V1_UNCHANGED", "PATTERN_ONLY", "仅形态"), ("V1_UNCHANGED", "STATE_ONLY", "固定状态过滤"),
             ("V1_UNCHANGED", "RECENT_ONLY", "原近期启停6例"), ("V2_SAMPLE_RELAXED", "RECENT_ONLY", "V2近期启停3例"),
             ("V1_UNCHANGED", "FULL", "原完整启停6/4"), ("V2_SAMPLE_RELAXED", "FULL", "V2完整启停3/2"),
             ("V1_UNCHANGED", "BUY_HOLD", "买入持有")]
    for variant, policy, label in specs:
        a = comp[(comp.variant == variant) & (comp.policy == policy)]
        b, s = a[a.cost == "BASE"].iloc[0], a[a.cost == "STRESS"].iloc[0]
        cycles = "持续持有" if policy == "BUY_HOLD" else str(int(s.completed_cycles))
        table.append(f"|{label}|{pct(b.net_cagr)}|{num(b.net_sharpe)}|{pct(s.net_cagr)}|{num(s.net_sharpe)}|{pct(s.max_drawdown)}|{cycles}|")
    write("研究结论.md", f"""
# 2021—2026样本门槛放宽V2：没有建立启停增量

已按用户要求把验证范围扩到2021-01-04至2026-09-16，并把最低近期成熟案例6降至3、同状态案例4降至2。仅改变这两项，成本、形态、回看窗口和盈利证据要求保持。结果是近期启停增加1笔亏损交易，完整启停仍0笔；这次适度放宽没有解决识别问题。完整机制的年化10%、夏普1.2目标仍未达成。

## 完整账户结果

初始20万元，{summary['sessions']}个交易日全部计入，包括空仓。2021年前已经成熟的资料只用于初始化当时资格；2021年之后的标签不能提前进入。2026年只到9月16日，不是完整年度。年度交易次数下限保持取消，回撤风险目标仍为10%。

{chr(10).join(table)}

基础/压力单边佣金万二/万四、滑点5/10基点，最低佣金5元；原始价格、股息应收和到账、T+1、整手、次日开盘、涨跌停延迟、终点卖出成本储备均沿用V1。买入持有的期末未卖不等于没有持仓。复合年化为负而算术均值定义的夏普略正，可以由波动拖累产生，不能混用两种年化口径。

仅形态在这段较长区间基础年化4.72%、夏普0.621；压力年化3.66%、夏普0.491，回撤11.19%。这与上一轮2024年8月起近期窗口的13.63%/1.185必须分别看，不能拼接或用较好区间覆盖较差区间。单个形态不必十年盈利，但当前这套机制在用户新指定完整区间没有达到目标。

## 放宽实际改变了什么

2021年以来42次确认，原规则仅形态完整账户40个周期。V2新增47条“某日某形态具备完整启用资格”的记录，但没有与新的形态确认在同一天出现，所以完整机制仍然不入场。不能在没有形态确认的资格日虚构机会。

42次确认中，{counts.get('STATE_INELIGIBLE', 0)}次首先被固定市场状态拒绝，{counts.get('RECENT_EDGE_UNCONFIRMED', 0)}次因近期盈利证据未确认，{counts.get('INSUFFICIENT_RECENT_EVENTS', 0)}次仍不足3个合格案例。这些是按固定判定顺序给出的首个原因，有些触发可能同时不满足多项条件；完整字段保留在relaxed/trigger_decisions.csv。

不要求固定市场状态的“仅近期启停”确实多出现一笔：{single.entry_date}买入、{single.exit_date}退出，形态为急跌后修复，压力净亏损{abs(single.net_pnl):,.2f}元。它使该比较账户的年化为-0.25%、夏普-0.388，而不能证明放宽后出现可靠优势。该信号在确认时属于UP，预设修复形态只允许DOWN或RANGE，因此完整机制仍拒绝。

本轮结论针对最低样本门槛这一个变化。剩余主要问题是盈利证据与状态映射能否及时确认，不只是最少样本数。不能根据已见2024年成功周期直接删除所有状态限制或把收益下界条件改成事后最好结果；若再改，应作为另一项明确假设独立固定。

## 收益与证据边界

仅形态压力账户最大一笔占完整周期净利润{concentration['largest_share']:.2%}，前三笔占{concentration['top3_share']:.2%}。它仍受少数周期影响，完整明细见profit_concentration.json。仅形态较买入持有改善并不自动证明启停机制有效。

原V1已经使用过这段历史；V2是在看到V1结果后由用户授权放宽。所有计算仍按当时成熟信息顺序进行，但属于已见历史上的后续诊断，不是新的独立验证。没有新行情下载、模型拟合、参数网格、独立前向样本或订单。新生成16个账户：12个同区间原规则与基准、4个V2账户。

V1原始结果和压缩包保留。本轮V2冻结失败，年化10%、夏普1.2、回撤10%的完整目标未达。0笔账户的夏普未定义，不用0或无穷大代替。当前行情仍截至9月16日，9月24日输出NO_VIEW/ABSTAIN，不形成当前仓位判断。

## 每日观察与复核

原Windows工作日16:10本地观察任务保留，新增并排记录V1的6/4与V2的3/2资格；形态识别、来源和时间窗相同。缺当天完整本地资料仍不生成前向观察，市场采集任务保持禁用。此次入口检查是NO_VIEW，前向天数0。资料齐全后快照另外保存decision_v2_min3_state2.csv，不覆盖V1记录。

只读复核入口为code/verify_sequential_patterns_relaxed_v2.py，传入--root解压目录；复算保存的资格、资金恒等式、日收益、T+1与指标，不新建账户或下载。原V1七项机制测试与四个真实历史截断核对记录在附带原包中。本轮只改两个样本条件，并逐项断言其余前序统计完全一致、新资格只能增加不能减少。
""")
    write("user_request.md", """
# V2用户原始补充

用户：如果继续放宽，21年到26年

澄清问题：你说的“继续放宽，21年到26年”，是否也要降低启停所需的样本门槛？

用户答复：扩大到2021—2026，并适度降低启停样本门槛。

代理事前说明：近期最低成熟案例6→3，同状态最低案例4→2；回看窗口、收益证据要求、形态、持有期和费用保持。验收年化10%、夏普1.2沿用用户本任务主请求，20万元和回撤风险目标10%沿用当前口径，年度次数下限已由用户取消。原“不用采集”保持。

此前完整主请求及V1工作保存在prior/V1_GPT_review.zip中，本轮没有覆盖V1。
""")
    write("00_README_FIRST.md", """
# 当前交付：2021—2026样本门槛放宽V2

先读研究结论.md和2021至2026账户对比.png，再读user_request.md、protocol.md。V2只把样本数量6/4改为3/2；没有放宽收益、风险、成本、状态映射或交易执行。

comparison.csv包含全部16条完整账户指标，annual.csv包含逐年结果。baseline是2021年起原规则对照，relaxed是V2，仅近期启停增加1笔亏损，完整启停仍0笔。relaxed/eligibility_changes.csv、trigger_decisions.csv保留启停变化和每次机会的判定；不能将47条新增资格记录当作47次交易。

inputs自含重新计算所需价格、股息、前序特征、完整形态信号、事件标签、普通时点对照和原资格。sources包含直接来源材料。baseline_freeze.json和relaxation_freeze.json分别固定两阶段输入与代码。前序统计除两个资格门槛外完全相同。

prior/V1_GPT_review.zip是经过SHA-256身份核对的原始完整V1快照，包含三类247个完整过程、87次确认、原36账户、机制测试、原用户请求和分红来源；它是旧快照，不可用旧近期指标替代本轮2021年起指标。当前ZIP根目录FILE_INDEX.csv是本包权威索引。

在解压目录安装requirements.txt后，用Python运行 `python code/verify_sequential_patterns_relaxed_v2.py --root .` 只读复算。本包结构核对和保存计算一致性不等于已完成外部GPT审阅、独立前向验证或达到收益目标。
""")
    write("01_GPT_REVIEW_PROMPT.md", """
请审阅本包2021—2026的V2样本门槛放宽试验，先读用户请求、协议和结论。

核查V2是否真的只把成熟案例6→3、同状态4→2；其他前序指标、成本、形态和执行是否一致。重点解释47条新增资格为何没有对应确认交易，以及唯一新增近期启停交易为何亏损。区分首个拒绝原因与全部同时不满足条件。

比较仅形态、固定状态、原启停和V2启停的完整账户；核查所有空仓日、T+1、股息应收到账、整手、最低佣金、滑点、涨跌停延迟和终点估值。不要把0笔的夏普写成0或把低回撤当收益目标达成。检查仅形态收益是否过度集中，2024年起近期结果与2021年起结果为何不同，不能挑最好片段。

这些历史已经用于V1及更早研究，V2不是未见验证。请给出有明确优先级、验证方式和停止条件的下一项研究假设：优先考虑状态识别迟滞和条件收益证据，而非机械继续降低样本数。不得依据已见成功交易直接挑新窗口、减费用或提高频次。用户已取消年度次数下限。

目标是510300与现金、20万元、净年化10%、夏普1.2、回撤风险目标10%；不用采集保持。请把本地可复核事实、机制推断、待验证方案分开，明确是否足以称为“当前有效”。不得自动扩张为券商、订单、Paper/Shadow或实盘权限。
""")
    write("EXCLUSIONS.md", """
# 包范围与排除

包含当前V2全部直接输入、16账户、资格和触发日记录、代码、冻结记录及完整原V1附包。没有重建全部历史原始下载，没有分钟订单簿或真实成交。当前数据至2026-09-16，独立前向观察0；不包含虚拟环境、Git目录、缓存或凭据。

本轮没有新增抽样或回归拟合；V1附包的区块抽样仅属于V1旧近期窗口，其零方差夏普差异行已作废，不能迁移到V2使用。V2数值结果冻结失败，不将归档或CRC通过写成策略通过或外部模型已审阅。
""")
    write("requirements.txt", "numpy\npandas\npyarrow\nmatplotlib\ntzdata")
    plot()
    for name in ("verify_sequential_patterns_relaxed_v2.py", "deliver_sequential_patterns_relaxed_v2.py", "observe_sequential_patterns_regime_v1.py"):
        shutil.copy2(ROOT / "scripts" / name, OUT / "code" / name)
    (OUT / "forward").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/forward/510300_sequential_patterns_regime_v1/latest_status.json", OUT / "forward/observer_check.json")
    shutil.copy2(PARENT / "forward/scheduler_receipt.json", OUT / "forward/scheduler_receipt.json")
    save_json(OUT / "forward/observer_revision.json", {"recorded_at": now(), "same_task_and_schedule": True,
              "task_name": "Codex-510300-Sequential-Patterns-Observe-V1", "versions_recorded": ["V1_6_4", "V2_3_2"],
              "workspace_script_sha256": digest(ROOT / "scripts/observe_sequential_patterns_regime_v1.py"),
              "v1_archive_not_changed": True, "collection_enabled": False, "current_observation": "NO_VIEW"})
    parent_archive = ROOT / "deliverables/510300_连续形态与状态启停_V1_GPT审阅_20260924.zip"
    expected = "909f7d43426b0de12bdade3e79136342573ee4297aa587eda6a8e9fd381a000d"
    assert digest(parent_archive) == expected
    (OUT / "prior").mkdir(exist_ok=True)
    shutil.copy2(parent_archive, OUT / "prior/V1_GPT_review.zip")
    with zipfile.ZipFile(parent_archive) as z:
        for item in z.namelist():
            if item.startswith("sources/"):
                target = OUT / item
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(item))
    save_json(OUT / "parent_identity.json", {"original_archive": str(parent_archive), "sha256": expected,
              "copy": "prior/V1_GPT_review.zip", "verified_before_expansion": True})
    mandate_file = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(mandate_file.read_text(encoding="utf-8"))
    mandate.update(current_round="510300_SEQUENTIAL_PATTERNS_2021_2026_V2",
                   current_protocol="reports/research/510300_sequential_patterns_2021_2026_v2/protocol.json",
                   latest_progress_receipt="reports/research/510300_sequential_patterns_2021_2026_v2/summary.json",
                   latest_user_instruction="将评价范围扩到2021—2026，并适度降低启停样本门槛；取消年度次数下限，不用采集保持。",
                   last_research_result="2021起仅形态压力年化3.66%、夏普0.491、回撤11.19%；数量门槛6/4降到3/2后近期启停多1笔亏损，完整启停仍0笔。目标未达。")
    save_json(mandate_file, mandate)
    save_json(OUT / "current_mandate_snapshot.json", mandate)
    save_json(ROOT / "config/510300_sequential_patterns_regime_v2.json", json.loads((OUT / "protocol.json").read_text(encoding="utf-8")))
    status_file = ROOT / "RESEARCH_STATUS.md"
    old = status_file.read_text(encoding="utf-8")
    if "510300_SEQUENTIAL_PATTERNS_2021_2026_V2" not in old:
        note = "> 2026-09-24 按用户追加要求完成 `510300_SEQUENTIAL_PATTERNS_2021_2026_V2`：评价2021-01-04至2026-09-16，最低近期案例6→3、同状态4→2，其他规则与成本不变。16个完整账户；仅形态压力年化3.66%、夏普0.491、回撤11.19%；仅近期启停新增1笔亏损，完整启停仍0笔。新增47条形态资格日没有对应确认信号，目标10%/1.2未达，V2冻结失败。原V1保留；本地16:10观察并排记录两版，不采集、前向0、当前NO_VIEW。见[本轮结论](reports/research/510300_sequential_patterns_2021_2026_v2/研究结论.md)。\n\n"
        status_file.write_text(note + old, encoding="utf-8")
    files = sorted([p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "FILE_INDEX.csv" and p.suffix != ".pyc"], key=lambda x: x.relative_to(OUT).as_posix())
    rows = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    index = buf.getvalue().encode("utf-8-sig")
    (OUT / "FILE_INDEX.csv").write_bytes(index)
    build = ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(build, "x", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for path, row in zip(files, rows):
            z.write(path, row["path"])
        z.writestr("FILE_INDEX.csv", index)
    extract = ROOT / "deliverables/verification" / ("sequential_patterns_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    extract.mkdir(parents=True, exist_ok=False)
    import hashlib
    with zipfile.ZipFile(build) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(rows) + 1
        assert set(z.namelist()) == {x["path"] for x in rows} | {"FILE_INDEX.csv"}
        for row in rows:
            blob = z.read(row["path"])
            assert len(blob) == row["bytes"] and hashlib.sha256(blob).hexdigest() == row["sha256"]
        z.extractall(extract)
    check = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(extract / "code/verify_sequential_patterns_relaxed_v2.py"), "--root", str(extract)],
                           cwd=extract, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    ZIP.with_suffix(".verification.txt").write_text(check.stdout + check.stderr, encoding="utf-8")
    if check.returncode:
        raise RuntimeError("新解压目录只读复算失败；building包保留。" + check.stdout + check.stderr)
    assert build.stat().st_size < 450000000
    build.replace(ZIP)
    receipt = {"delivered_at": now(), "path": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": digest(ZIP),
               "zip_members": len(rows) + 1, "indexed_members": len(rows), "crc_index_size_hash": "PASS",
               "fresh_extraction_saved_recomputation": "PASS", "verification_output": check.stdout,
               "includes_original_v1_archive": True, "parent_sha256": expected,
               "new_accounts_in_verification": 0, "new_fits": 0, "new_draws": 0, "downloads": 0,
               "goal_achieved": False, "current_view": "NO_VIEW", "external_review_completed": False}
    save_json(ZIP.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
