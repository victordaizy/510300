"""从冻结的 EPU 研究结果生成中文说明、图表与可重算审阅包。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

OUT = ROOT / "reports/research/510300_epu_vintage_increment_v1"
ZIP = ROOT / "deliverables/510300_EPU历史版本信息增量_V1_GPT审阅_20260913.zip"
NAMES = {"M0": "ETF价格基线", "M1": "加入EPU", "VOL_ONLY": "仅波动控制", "BUY_HOLD": "买入持有"}


def text_once(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def plot_saved() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [1.6, 1, 1]})
    colors = {"M0": "#426987", "M1": "#be4f36", "VOL_ONLY": "#31866d", "BUY_HOLD": "#7b7e86"}
    for model in NAMES:
        frame = pd.read_parquet(OUT / "accounts" / ("STRESS_" + model + "_ledger.parquet"))
        axes[0].plot(frame.date, frame.equity / 200000 * 100, color=colors[model], label=NAMES[model], linewidth=1.5)
        if model != "BUY_HOLD":
            axes[2].plot(frame.date, frame.exposure * 100, color=colors[model], linewidth=1.05)
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    predictions = predictions.loc[predictions.label_status.eq("MATURE")]
    axes[1].step(predictions.entry_date, predictions.epu_latest_value, where="post", color="#70528c", linewidth=1.4)
    missing = predictions.loc[predictions.prediction_status.ne("PASS")]
    axes[1].scatter(missing.entry_date, missing.epu_latest_value, marker="x", color="#be4f36", s=55, zorder=3, label="统计月过旧，NO_VIEW")
    axes[0].set_title("EPU历史版本：年化接近目标，夏普与增量证据仍不足", loc="left", fontproperties=font, fontsize=17)
    for ax, label in zip(axes, ["压力账户净值（初始=100）", "决策前已可见的EPU值", "实际510300持有比例（%）"]):
        ax.set_ylabel(label, fontproperties=font, fontsize=10)
        ax.grid(alpha=.17)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].legend(prop=font, ncol=2, loc="upper left", frameon=False)
    axes[1].legend(prop=font, loc="upper right", frameon=False)
    axes[2].set_ylim(-3, 103)
    axes[2].set_xlabel("2023-01-03至2026-08-03；冻结月度规则与历史报价假设，未获独立未来验证", fontproperties=font, fontsize=10)
    fig.savefig(OUT / "EPU压力账户与历史信息.png", dpi=160, facecolor="white")
    plt.close(fig)


def report(result: dict) -> str:
    accounts = ["| 账户 | 费用 | 净复合年化 | 净夏普 | 最大回撤 | 平均持有比例 |", "|---|---|---:|---:|---:|---:|"]
    for a in result["accounts"]:
        accounts.append(f"| {NAMES[a['model']]} | {a['cost']} | {a['annualized_return']:.2%} | {a['net_sharpe']:.3f} | {a['max_drawdown']:.2%} | {a['mean_exposure']:.2%} |")
    prediction_rows = ["| 时期 | 有效配对月 | M1相对M0的预测误差改善 |", "|---|---:|---:|"]
    for row in pd.read_csv(OUT / "prediction_comparison.csv").itertuples():
        prediction_rows.append(f"| {row.era} | {row.paired_months} | {row.relative_mse_improvement:+.2%} |")
    return f"""# EPU历史版本月度增量：有局部改善，未建立可靠优势

本轮完成77个免费历史版本的取得和核对、82个月度样本、88次拟合、44次预测及8个完整账户。EPU模型基础费用年化10.031%、夏普0.944；压力费用年化9.687%、夏普0.916。只交易510300和现金、成本后年化10%及夏普1.2的共同目标尚未实现。

终态：`{result['status']}`。这一固定用途已按事前规则结案；不得为了接近10%而增加仓位、换日期、反转条件、改变EPU窗口或拼接旧策略。局部改善可以如实保留，但不转成待交易信号。

## 为什么是这项新信息

原用户先要求检查IF持仓建议，随后明确允许其他免费有效信息。IF持仓与A/H溢价的固定用途已经分别结案。本轮使用两家大陆报纸的经济、政策、不确定性词组频率指数，与ETF价格、期货持仓、A/H同股定价有实质区别。定向检查现有方案未找到相同的EPU历史版本增量试验。不是重新开启原有价格规则筛选。

## 真实发布版本的取得与边界

通过[ALFRED公开下载表单](https://alfred.stlouisfed.org/series/downloaddata?seid=CHNMAINLANDEPU)取得全部77个版本，2010-01至2026-08共200个统计月、202行实时区间记录。最早版本2019-10-09，最近2026-09-09。与另外下载的2022-09-09、2023-01-06版本矩阵合计308个值完全一致，最大误差0；保留两处实际修订和全部原始响应、表单及请求参数。

两处修订为2022-08的260.9在2022-11-03修订为260.880967，2022-11的448.6在2023-01-06修订为448.588846。代码按每次决定可见的版本选择，不把修订值填回早先特征。2020-01一次补发2019-09至12月；2025-01-17一次补发2024-10至12月。因此不能固定假定每月某一天必然已经公布上月数据。

最早一批历史值不是早年的逐月发布时间证明。只从2019-11开始形成月度交易原点，2010年以来回溯数据仅用于2019年以后已可见的过去12月尺度，不当作2019年前训练或评价信号。[原作者方法](https://www.policyuncertainty.com/china_monthly.html)的大陆两报构造使用2000—2018年的标准化，进一步要求明确后见数据的边界。

版本日期按芝加哥当地零点加两个自然日后再视为可用，转换为上海时区，每月首内地交易日09:00判断，09:30执行。这个缓冲适用于本轮历史研究，不是过去逐次网页到达凭据。[ALFRED说明](https://alfred.stlouisfed.org/help)指出，发布日期可能取自原来源、供应商或首次进入FRED日期；本次下载收到数据不能证明过去本站同一时刻可下载。

2025-01-02只能看到2024-09统计月，超过事前三个月滞后限额，该月两个配对模型均为NO_VIEW并在模拟政策中取现金。2025-01-17补发的数据没有被用于1月2日判断。该月市场结果保留在完整账户时序，不作为有信息的配对预测。

## 固定模型及月度预测

从2019-11按月扩展训练，2023-01开始评价，第一次有35个已成熟有效训练月。M0为ETF20/60日含分红收益及20日方差对数；M1只加EPU相对前12月几何均值和一月对数变化。标准化、±5截断、岭罚0.1、同样本配对拟合。标签退出日必须早于模型前一内地收盘原点；同月未来变化不能影响本月预测。

44个评价月中43个完整月，1个截至行情末日尚未完成的8月标签；完整月中1个NO_VIEW，实际配对42个月，使用39个不同的最新数值版本。没有把日线867行当成867份新闻信息。误差检验以连续三个日历月循环分块2000次，缺失月份保留原位置。

{chr(10).join(prediction_rows)}

总体误差改善1.755%，方向准确率从52.38%到57.14%；改善95%区间为[-0.0005432543, 0.0008427784]，跨零。2023年和2025年预测误差反而变差，三个完整年仅一个年改善。

M1判断参与的20个月，独立全额持有月标签均值+1.575%；判断回避的22个月均值-0.058%。这是有利的历史条件差，但参与月份的最差持有期标记收益仍达-17.094%。这些是固定持有标签统计，不是账户年化，也未证明未来能稳定挑出上涨或避开下跌。

## 完整账户与真正的对照

所有账户均从2023-01-03至2026-08-03，共867个账本日；终点为最后一个完整月度周期的下一月首开盘，其后9个行情日不交易。没有选择最佳结束日期。每个账户20万元、仅510300和人民币现金，无杠杆、整百份、T+1、方向涨跌停和现金约束。分红登记、除息应收和现金派发分开处理。

M0/M1只有净收益预测为正才参与，按min(1,10%/ETF市场年化波动)确定月首比例。代码的RV20是方差，已在冻结前纠正并测试sqrt(RV20×242)的标准差换算。VOL_ONLY不预测方向，每个月直接采用同一波动预算；BUY_HOLD仅起点买入、终点卖出。波动预算是事前估计，不保证实际波动恰为10%。

{chr(10).join(accounts)}

基础佣金每向万二、至少5元、滑点5bp；压力万四、至少5元、滑点10bp，价格按0.001向不利方向取整。现金和夏普基准收益假设零，年交易日242。压力M1最终账户278542.01元，29次成交，平均持有32.96%，最大回撤17.26%。这是报价执行假设下的完整资金账，不是实际成交或实盘收益。

M1相对M0压力年化算术收益差+5.756个百分点，三个月区块95%区间[-7.834,+24.830]个百分点。相对VOL_ONLY则为-1.772个百分点，区间[-10.596,+5.123]个百分点。两者都未确认正增量，且这些差不是控制实际敞口后的回归alpha。

2024年M1压力账户年化33.59%，2023年为-6.22%，2025年8.06%，2026不完整年按同口径年化4.96%。2024年良好表现与其他时期的区别保留；不删去2023，不只展示2024，不把2026部分年度说成已经实现全年收益。仅波动控制同期年化11.30%、夏普0.868，说明本轮信息并未带来相对这个简单对照的明确收益优势。

## 验证、结案和下一步

冻结时间{result['freeze_created_at']}，完成时间{result['completed_at']}。10项冻结前合成测试通过；保存重算检查82个月度版本、88个模型、44个预测和8个账本，最大预测误差0；未新拟合、未新开账户、未重新抽样或联网。图由保存账本生成。

预测与账户两条事前候选准入路径均失败，故本次EPU水平/一月变化的固定用途关闭。结论限于这一用途，不泛化为所有新闻信息无效。即使基础年化略过10%，夏普仍缺0.256，压力口径两个目标均未到；增加仓位不自动改善夏普。

后续优先继续检查有明确事前发布时间的非价格供给信息。限售解禁、股东减持计划等仅作为去重和免费来源可得性候选，尚未准入、未构建因子、未试算收益；如果已有相同终态或缺少历史公告时刻，停止该候选，不重启旧家族。已研究的两融、基金份额、宏观发布和旧价格层不能仅换名称再次启动。

本轮按PROGRESS登记：增加了一个完整、可复算的非价格试验和否证，整体目标保持未实现。自包含审阅ZIP不是外部审阅；没有上传、没有订单或券商连接，也没有另做安全审计。
"""


def main() -> None:
    require(not ZIP.exists(), "已有交付不可覆盖")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verified = json.loads((OUT / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(result["status"] == "REJECTED_FROZEN_NO_RELIABLE_EPU_INCREMENT" and verified["accounts_checked"] == 8, "研究终态或保存检查不符")
    body = report(result)
    text_once(OUT / "研究结论与下一步.md", body)
    plot_saved()
    prompt = """# 可复制的GPT审阅提示

请用中文审阅本包，目标仍为仅510300与人民币现金，成本后复合年化至少10%、净夏普至少1.2，且有独立证据。先读00_README_FIRST.md和02_研究结论与下一步.md，再查冻结协议、输入、模型、账户与回执。

本次新信息为大陆两报EPU指数的ALFRED历史版本。只从2019年公开版本之后构造月度样本；M0为ETF20/60日收益和风险，M1加两个EPU特征。基础年化10.031%、夏普0.944；压力9.687%、0.916。43完整评价月、42有效配对月、39个最新数值版本。整体MSE改善1.755%但跨期不稳定；相对价格基线和仅波动控制的经济区间均跨零。本用途已按冻结门拒绝，请勿因接近目标而反向、换窗、换区间或增加倍率救援。

请交付：
1. 研究质量结论以及P0/P1/P2具体问题、文件和字段。重点核查版本区间、修订延迟、芝加哥/上海时区、首个公开版本与后见历史、按月对齐、训练标签严格成熟。
2. 核查2025-01 NO_VIEW、批量补发、39个版本和42配对月能支持多大判断；分块应保留缺失月份、抽取连续月份而非把每日重复观测当独立新闻。
3. 核查标签从成交后开始、月首退出、T+1、整数份额、方向涨跌停、分红权益与应收现金、日财富恒等式。特别区分RV20方差与年化标准差，区别波动目标与实际波动。
4. 同时评价M0、M1、VOL_ONLY和BUY_HOLD。预测改善或方向正确率不能替代完整账户，也不要把不同实际敞口的收益差视为已证明alpha。
5. 评价2024年表现与其他完整年度、样本长度、宽区间和项目重复研究造成的选择偏差。包完整性不是独立验证，局部改善不等于达标。
6. 给出唯一下一优先方向和最低成本的验证顺序：先核对是否已做，再取得免费直接来源和历史可得时点，再冻结一个小实验、完整账户和明确停止条件。事前公布供给信息仅为待核查候选，不能假定限售解禁或减持未研究，也不能自动重启旧两融、基金份额、宏观或价格失败用途。
7. 明确离10%与1.2分别缺哪些实证，不降低目标，不保证最终必然可实现。没有外部上传、实盘或订单授权。
"""
    text_once(OUT / "GPT审阅提示.md", prompt)
    files = {}

    def add(path: Path) -> None:
        require(path.is_file(), "必要文件缺失：" + str(path))
        files[path.relative_to(ROOT).as_posix()] = path

    for path in OUT.rglob("*"):
        if path.is_file():
            add(path)
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["protected_files"] + manifest["input_snapshots"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结文件变化")
        add(ROOT / item["path"])
    cfg = json.loads((ROOT / "config/510300_epu_vintage_increment_v1.json").read_text(encoding="utf-8"))
    for rel in cfg["inputs"].values():
        files[rel] = OUT / "frozen_inputs" / rel
    source_dir = ROOT / "data/raw/market/510300_epu_vintages_source_v1/20260913T231935_0800"
    for path in source_dir.iterdir():
        if path.is_file():
            add(path)
    for rel in ["scripts/build_epu_vintage_review_20260913.py", "scripts/retrieve_510300_epu_public_vintages_v1.py",
                "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md", "docs/510300_CURRENT_RESEARCH_DISPOSITION_20260913.md",
                "config/510300_if_open_interest_increment_v1.json", "config/510300_ah_premium_increment_v1.json",
                "reports/research/510300_if_open_interest_increment_v1/result.json", "reports/research/510300_ah_premium_increment_v1/result.json",
                "data/raw/r6/510300_daily.parquet", "data/reference/510300_downside_risk_price_corrections_v1.csv"]:
        add(ROOT / rel)
    coverage = json.loads((OUT / "frozen_inputs" / cfg["inputs"]["dividend_coverage"]).read_text(encoding="utf-8"))

    def add_sources(value) -> None:
        if isinstance(value, dict):
            if value.get("saved_file"):
                path = ROOT / value["saved_file"]
                if value.get("sha256"):
                    require(digest(path) == value["sha256"], "分红来源哈希不符")
                add(path)
            for child in value.values():
                add_sources(child)
        elif isinstance(value, list):
            for child in value:
                add_sources(child)

    add_sources(coverage)
    readme = """# EPU历史版本信息增量V1：阅读导航

本轮已完成并拒绝这一固定用途：压力年化9.687%、净夏普0.916，整体目标尚未实现。请先读02_研究结论与下一步.md、01_GPT_REVIEW_PROMPT.md，再查docs/510300_EPU_VINTAGE_INCREMENT_V1_PROTOCOL.md和配置。

主要证据在reports/research/510300_epu_vintage_increment_v1：USER_REQUEST.md、deduplication.json、data_preflight.json、freeze_manifest.json、models.json、predictions.parquet、accounts完整账本及saved_verification_receipt.json。图为EPU压力账户与历史信息.png。原始EPU下载与308个历史值交叉核对在data/raw/market/510300_epu_vintages_source_v1/20260913T231935_0800。

本包包含本轮全部直接输入及独立冻结快照、代码、测试、全部模型与账户、原始EPU公开表单与ZIP、官方方法、分红直接凭据、价格原始值与更正。旧IF/AH只包含去重和结果背景，不提供整个旧研究闭包，也不要求重跑它们。排除虚拟环境、缓存和全项目其余研究。

FILE_INDEX.csv列出除索引自身以外所有成员的字节数与SHA-256；外部同名.delivery.json记录最终ZIP哈希。包CRC、哈希、索引和解压后核对只证明所述保存材料一致，不证明科学有效或已获外部GPT审阅。

Windows复查：在解压目录准备Python3.13并安装REQUIREMENTS_REVIEW.txt依赖，然后执行`python scripts/run_510300_epu_vintage_increment_v1.py verify`。该命令从已存模型、预测、历史版本、账本和月份抽样重算，不训练、不新建账户、不重新抽样、不联网。合成测试命令是`python -m pytest tests/test_epu_vintage_increment_v1.py -q`。

run已经消耗唯一机会，禁止删claim或改冻结文件重跑。retrieve_510300_epu_public_vintages_v1.py是此次公开表单请求的可复现来源说明代码，未在本轮追加运行；它只可创建新的时间戳目录，不改变本包冻结输入，也不表示已启动未来每日任务。本包未上传、未获外部审阅，不包含交易指令。
"""
    dependencies = ["numpy", "pandas", "pyarrow", "pytest", "tzdata", "requests", "beautifulsoup4", "matplotlib"]
    requirement = "\n".join(name + "==" + importlib.metadata.version(name) for name in dependencies) + "\n"
    virtual = {"00_README_FIRST.md": readme.encode("utf-8"), "01_GPT_REVIEW_PROMPT.md": prompt.encode("utf-8"),
               "02_研究结论与下一步.md": body.encode("utf-8"), "REQUIREMENTS_REVIEW.txt": requirement.encode("utf-8")}
    entries = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for name, path in files.items()]
    entries.extend({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in virtual.items())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda x: x["path"]))
    virtual["FILE_INDEX.csv"] = buffer.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    require(not building.exists(), "已有构建残留，不能覆盖")
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        for name, raw in sorted(virtual.items()):
            archive.writestr(name, raw)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "重复成员或ZIP CRC错误")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) - {"FILE_INDEX.csv"} == {x["path"] for x in index}, "索引覆盖不全")
        for item in index:
            raw = archive.read(item["path"])
            require(len(raw) == int(item["bytes"]) and hashlib.sha256(raw).hexdigest() == item["sha256"], "成员与索引不符")
    os.replace(building, ZIP)
    delivery = {"status": "PASS_ZIP_CRC_INDEX_HASH_AND_DECLARED_COVERAGE", "created_at": now(), "zip": ZIP.relative_to(ROOT).as_posix(),
                "bytes": ZIP.stat().st_size, "sha256": digest(ZIP), "members": len(names), "indexed_files": len(index),
                "models_in_study": 88, "accounts_in_study": 8, "builder_new_models": 0, "builder_new_accounts": 0,
                "security_audit": False, "external_review": "NOT_PERFORMED", "upload_performed": False}
    write_json(ZIP.with_suffix(".delivery.json"), delivery, exclusive=True)
    print(json.dumps(delivery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
