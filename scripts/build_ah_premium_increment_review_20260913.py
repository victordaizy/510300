"""只使用保存结果，生成A/H增量结案报告、图和自包含GPT审阅包。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_ah_premium_increment_v1"
ZIP = ROOT / "deliverables/510300_AH溢价信息增量_V1_GPT审阅_20260913.zip"


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def create_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1, 1]}, layout="constrained")
    colors = {"M0": "#487c9b", "M1": "#d35d36", "BUY_HOLD": "#5b636b"}
    names = {"M0": "M0：ETF与恒指信息", "M1": "M1：增加A/H溢价", "BUY_HOLD": "同期买入持有"}
    for model in ("M0", "M1", "BUY_HOLD"):
        ledger = pd.read_parquet(OUT / "accounts" / ("STRESS_" + model + "_ledger.parquet"))
        axes[0].plot(ledger.date, ledger.equity / 200000 * 100, label=names[model], color=colors[model], linewidth=1.5)
        if model != "BUY_HOLD":
            axes[2].plot(ledger.date, ledger.exposure * 100, label=names[model], color=colors[model], linewidth=1)
    f = pd.read_parquet(OUT / "features.parquet")
    f = f.loc[f.date.between("2024-01-02", "2026-08-11")]
    axes[1].plot(f.date, f.ah_close, color="#624a88", linewidth=1.2)
    axes[0].set_title("A/H溢价增量检验：压力费用下的固定历史账户", fontproperties=font, fontsize=17, loc="left")
    for ax, title in zip(axes, ["账户净值（初始=100）", "决定时已对齐的A/H溢价指数", "实际持有510300的市值占比（%）"]):
        ax.set_ylabel(title, fontproperties=font, fontsize=10)
        ax.grid(alpha=.16)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].legend(prop=font, loc="upper left", frameon=False)
    axes[2].set_ylim(-3, 103)
    axes[2].set_xlabel("2024-01-02至2026-08-11；历史报价成交假设，非实际成交或独立未来验证", fontproperties=font, fontsize=10)
    fig.savefig(OUT / "压力账户与持有比例.png", dpi=160, facecolor="white")
    plt.close(fig)


def report_text(result: dict, comparisons: pd.DataFrame) -> str:
    table = ["| 账户 | 费用 | 净复合年化 | 净夏普 | 最大回撤 | 平均持有比例 |", "|---|---|---:|---:|---:|---:|"]
    for a in result["accounts"]:
        label = {"M0": "ETF+恒指基线", "M1": "增加A/H信息", "BUY_HOLD": "同期买入持有"}[a["model"]]
        table.append(f"| {label} | {a['cost']} | {a['annualized_return']:.2%} | {a['net_sharpe']:.3f} | {a['max_drawdown']:.2%} | {a['mean_exposure']:.2%} |")
    pred = ["| 评价入场时期 | 成熟原点 | M1相对M0的均方误差改善 |", "|---|---:|---:|"]
    for row in comparisons.itertuples():
        pred.append(f"| {row.group} | {row.paired_origins} | {row.relative_mse_improvement:+.3%} |")
    e = result["evaluation"]
    low, high = e["paired_mse_improvement_95pct_ci"]
    return f"""# A/H同股跨市场溢价：未确认510300择时增量

已经完成数据取得、时钟处理、事前冻结、64次模型拟合、629个成熟配对预测及6个完整账户。本轮未达到仅交易510300、成本后年化10%、夏普1.2的目标。

固定用途的终态为`{result['status']}`。M1整体预测均方误差较M0增大2.612%，三个固定时期均未改善。压力成本M1账户年化2.228%、净夏普0.420；本次用途结案，不追加窗口、阈值、反向、仓位倍率或第三个过滤器。

## 为什么选这一项以及它是什么

本次用户进一步允许使用其他免费有效信息。在当前配置和定向研究记录中，既有海外信息分支使用恒生指数等整体价格，但没有发现同股A/H溢价的相同增量检验。A/H指数由同一批两地上市公司的A股和H股定价构成，与直接比较两个不同成份宽基指数不同。官方指数还受汇率、成份及权重变化影响，不能等同资金净流向或无风险套利。

M0控制ETF一日、五日、二十日含分红收益、20日风险以及恒指5/20个内地决定原点的收益；M1只增加ln(HSAHP/100)和五原点对数变化。通用标签、岭回归、完整账户计算复用已测试代码，IF数据和旧策略没有进入本轮，也没有修改其拒绝记录。

## 免费来源与明确局限

恒生官网公开图表保存1266条A/H日线，2021-09-10至2026-09-11。官方指数身份为01044.00。先将毫秒时间戳解释为UTC，再转香港时区取日期；直接取UTC日期会错一天。修正后46个近期共同日期与新浪完全一致，2026-08-31的123.53与官方事实表一致。

新浪仅返回46个近期日期；东方财富环境代理、直接网络及文档参数请求未取得数据，腾讯出现TLS连接错误，新浪旧路径404，Investing请求403；这些失败响应或异常回执均保留。恒生的通用公开dailyClose接口本次返回data=null；真正采用的是公开五年chart.json。未绕过TLS、认证或付费限制，也未购买完整历史。

训练从2021-10-08起扩展，第一次更新有538条成熟样本。根据免费数据长度预先定为2024-01-02开始评价，ETF标签沿用2026-08-14截止；没有覆盖2020—2021完整牛市或2015风险阶段。样本期选择发生在计算本轮标签和结果之前，但这些市场历史早已被项目观察，不能作为全新独立留出样本。

外部日线按“下一自然日08:00已可得”假设，在下一内地交易日09:00决定并使用09:30开盘报价。源日期最大落后A/H四自然日、恒指六自然日，没有超过固定7日限额。官网和供应商的逐日首次公开时刻仍未证明；本次HTTP接收时间仅证明本次收到，不能充当2021年以来的当时回执。

## 预测与完整账户结果

每个入场月首用完全相同的成熟行训练M0/M1；退出日必须严格早于月首原点日期。固定标准化、clip±5、岭惩罚0.1，截距不罚。32次配对更新、64次拟合，634次预测中629次标签成熟、5次截止删失。五日标签从下一开盘开始，不包含之前的跳空，含真实登记分红和基础费用。

{chr(10).join(pred)}

整体M0−M1均方误差差为{e['overall']['mse_improvement']:.10g}；20交易日、2000次配对循环区块95%区间为[{low:.10g}, {high:.10g}]，跨零。629条重叠标签不是629次独立交易，不能用标签均值直接年化。

M0与M1均为独立20万元连续现金账户，每5个交易日调整，净预测为正才参与，权重上限为min(1,10%/ETF市场年化波动)。买入持有只作同时段对照。以下均从2024-01-02至2026-08-11的631个账本日；共同终点是事前固定的最后完整五日格点，其后3日不进入账户，未选择最优终点。

{chr(10).join(table)}

基础每向佣金0.02%、最低5元、滑点0.05%；压力0.04%、最低5元、滑点0.10%，报价按0.001不利取整、100份整数、现金约束和T+1。现金利息与夏普基准均假设0，年化242日。分红登记、除息应收及付款分别入账。账户财富守恒；这些为历史报价假设，不是实际成交。

压力下M1比M0的年化算术日收益差约+0.978个百分点，但配对区块95%区间年化表达约[-3.451,+5.580]个百分点，不能确认正增量。M1平均持有比例19.54%，高于M0的12.12%；收益差并非已经证明为去除敞口差异后的信息收益。预测误差各时期均变差，也不支持用这点账户差值保留信号。

买入持有在这个特定区间年化14.764%、夏普0.846：年化跨过10%，夏普仍未达到1.2。新模型大幅减少持有并同时降低了收益，这尚未建立高质量的上涨参与和下跌回避。不能用降低风险这一点宣称完成目标，也不能通过提高倍率把0.420的夏普自动变成1.2。

图见`压力账户与持有比例.png`。图只是上述保存账本的展示，不生成新账户或变体。

## 已完成检查与资源决定

冻结于{result['freeze_created_at']}，完成于{result['completed_at']}。冻结前9项合成测试通过；保存核对64个模型、634次预测、6个完整账户及两组固定区块统计通过，最大预测重算误差0。核对没有新训练、新账户、新随机抽样或网络请求。

本次A/H水平与五日变化的固定用途关闭；这不证明全部A/H相关信息在所有模型、时期和用途下一概无效。下一项研究应继续寻找与已试信息有实质差异、免费且能在下一次成交前获得的观测，优先核查非价格信息；先去重及来源可得性，再固定一个实验。具体下一来源尚未确定、未运行，不列为已有成果。本轮不增加永久自动化或新的实盘信号。

自包含ZIP包含原始用户授权、冻结方案与代码、完整源数据与失败回执、模型预测、基础和压力账户、原始分红凭据、检查记录和可复制GPT审阅提示。只做必要数值及包CRC/哈希/索引检查，没有另做安全审计；没有上传、外部GPT审阅或订单。
"""


def main() -> None:
    require(not ZIP.exists(), "已交付包不能覆盖")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(result["status"] == "REJECTED_FROZEN_NO_RELIABLE_AH_INCREMENT", "交付器适用终态不符")
    require(verification["accounts_checked"] == 6 and verification["new_accounts"] == 0, "保存核对未完成")
    report = report_text(result, pd.read_csv(OUT / "prediction_comparison.csv"))
    write_text(OUT / "研究结论与下一步.md", report)
    create_figure()
    prompt = """# 可直接复制给GPT的审阅提示

最终目标：只交易510300和人民币现金，实现成本后复合年化至少10%、净夏普至少1.2。请基于包内原始证据，用中文评价本轮质量，并给出下一阶段唯一最高优先事项。

先读00_README_FIRST.md、02_研究结论与下一步.md和冻结协议，再查原始来源、模型预测、完整账户及回执。此轮不是IF持仓重试：IF用途已结案，新的A/H信息来自官网免费五年图表；M0含ETF及恒指，M1只增A/H水平与五日变化。整体预测误差变差2.612%，三个时期均无改善；M1压力年化2.228%、夏普0.420，已按固定门拒绝。请勿把“有一条账户略高”当确认信息增量。

请逐项回答：
1. 提出P0/P1/P2具体问题与文件/字段；重点核查香港时区日期、两地节假日、假设的下一日08:00可得时钟、训练标签成熟、同模型同日期及未来数据隔离。
2. 检查五年公开数据导致2024开始评价的局限、训练期以2022—2023为主的适用范围、指数换样/汇率与定价混合含义，以及只有46日双源交叉核对能证明什么。
3. 检查20万元真实现金账本的费用、分红应收、整数份额、T+1、五日格点与共同终点。区分相同仓位规则与相同实际风险敞口；不要将M1比M0收益略高误称已证明alpha。
4. 检查MSE与完整账户两层结果、20日配对区块区间和三个时期。区间未处理项目多次研究选择偏差，历史研究不等于独立未来证据。
5. 对本次用途是否应停止提出结论。不得凭本结果反转、改窗口、选择年份、提高倍率、拼入旧策略或增加第三过滤器救援。
6. 下一步只选一项具有实质新增信息的免费来源，优先考虑非价格观测；先列与现有失败方向的差异，再给出最小数据合同、首次可得时钟、固定有限候选、完整账户检验、验收与停止条件。来源或历史覆盖不明时写NO_VIEW，提案写NOT_RUN。
7. 解释离10%与1.2分别还缺什么证据，不承诺任何方法必然达标。本包未获外部审阅，无券商/订单授权。
"""
    write_text(OUT / "GPT审阅提示.md", prompt)
    files = {}

    def add(path: Path) -> None:
        require(path.is_file(), "包内必要文件不存在：" + str(path))
        files[path.relative_to(ROOT).as_posix()] = path

    for path in OUT.rglob("*"):
        if path.is_file():
            add(path)
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["protected_files"] + manifest["input_snapshots"]:
        path = ROOT / item["path"]
        require(digest(path) == item["sha256"], "冻结文件变更：" + item["path"])
        add(path)
    cfg = json.loads((ROOT / "config/510300_ah_premium_increment_v1.json").read_text(encoding="utf-8"))
    for rel in cfg["inputs"].values():
        files[rel] = OUT / "frozen_inputs" / rel
    source_dir = ROOT / "data/raw/market/510300_ah_premium_source_v1/20260913T223902_0800"
    for path in source_dir.iterdir():
        if path.is_file():
            add(path)
    dedup = json.loads((OUT / "deduplication.json").read_text(encoding="utf-8"))
    for item in dedup["evidence"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "去重背景证据漂移")
        add(ROOT / item["path"])
    for rel in ["scripts/build_ah_premium_increment_review_20260913.py", "scripts/retrieve_510300_ah_official_source_snapshot_v1.py",
                "scripts/collect_510300_ah_premium_sources_v1.py", "scripts/collect_510300_cross_market_chart_ml_v1.py",
                "reports/research/510300_overnight_global_information_v1/result.json",
                "reports/research/510300_if_open_interest_increment_v1/研究结论与下一步.md",
                "reports/research/510300_if_open_interest_increment_v1/USER_PROPOSAL.txt",
                "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
                "data/raw/r6/510300_daily.parquet", "data/reference/510300_downside_risk_price_corrections_v1.csv"]:
        add(ROOT / rel)
    coverage = json.loads((OUT / "frozen_inputs" / cfg["inputs"]["dividend_coverage"]).read_text(encoding="utf-8"))

    def add_saved(value):
        if isinstance(value, dict):
            if value.get("saved_file"):
                path = ROOT / value["saved_file"]
                if value.get("sha256"):
                    require(digest(path) == value["sha256"], "分红直接凭据哈希不符")
                add(path)
            for child in value.values():
                add_saved(child)
        elif isinstance(value, list):
            for child in value:
                add_saved(child)

    add_saved(coverage)
    readme = """# A/H信息增量V1：先读这里

本轮已完成并拒绝：未确认新信息增量，压力账户年化2.228%、夏普0.420，10%/1.2尚未实现。

阅读顺序：02_研究结论与下一步.md → 01_GPT_REVIEW_PROMPT.md → reports/research/510300_ah_premium_increment_v1/USER_REQUEST.md → docs/510300_AH_PREMIUM_INCREMENT_V1_PROTOCOL.md及config同名配置 → 数据预检/冻结/模型预测/账户/核对回执。图在该研究目录的压力账户与持有比例.png。

本包包括本轮全部直接数据与代码、模型、完整账户、原始A/H公开网页数据和来源失败回执、14次分红凭据、原价格与三处更正、必要近邻背景。旧IF及全球信息资料仅供去重背景，不包含其完整旧数据闭包，也不要求重跑。排除整个项目的其余研究、虚拟环境、缓存和机器配置。

根FILE_INDEX.csv索引除自身外的每个成员，记录字节数和SHA-256。同名外部.delivery.json记录ZIP最终哈希；结构通过不等于科学有效或外部GPT审阅。本次没有上传、外部审阅或安全审计。

Windows核对：在解压目录准备Python3.13和REQUIREMENTS_REVIEW.txt列出的依赖，然后执行`python scripts/run_510300_ah_premium_increment_v1.py verify`。该模式只检查保存模型/预测/账本/抽样，不拟合、不下载、不新增账户。合成测试命令为`python -m pytest tests/test_ah_premium_increment_v1.py -q`。

run已使用一次机会，run_claim.json必须保留；禁止删除claim、修改冻结输入或调参重跑。本包的采集脚本是来源说明，不要运行它们覆盖冻结输入。retrieve_510300_ah_official_source_snapshot_v1.py仅能另存新时间戳目录，存在脚本不代表已启动每日采集。图与报告根据保存结果生成，没有新增策略变体。
"""
    requirements = "\n".join(name + "==" + importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "pytest", "tzdata", "requests"]) + "\n"
    virtual = {"00_README_FIRST.md": readme.encode("utf-8"), "01_GPT_REVIEW_PROMPT.md": prompt.encode("utf-8"),
               "02_研究结论与下一步.md": report.encode("utf-8"), "REQUIREMENTS_REVIEW.txt": requirements.encode("utf-8")}
    entries = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(files.items())]
    entries.extend({"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in sorted(virtual.items()))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda item: item["path"]))
    virtual["FILE_INDEX.csv"] = buffer.getvalue().encode("utf-8-sig")
    temporary = ZIP.with_suffix(".building.zip")
    require(not temporary.exists(), "存在未完成构建，不能静默覆盖")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        for name, value in sorted(virtual.items()):
            archive.writestr(name, value)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "成员重复")
        require(archive.testzip() is None, "ZIP CRC错误")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) - {"FILE_INDEX.csv"} == {item["path"] for item in index}, "索引覆盖错误")
        for item in index:
            value = archive.read(item["path"])
            require(len(value) == int(item["bytes"]) and hashlib.sha256(value).hexdigest() == item["sha256"], "压缩文件与索引不符")
    os.replace(temporary, ZIP)
    delivery = {"status": "PASS_ZIP_CRC_INDEX_HASH_AND_DECLARED_COVERAGE", "created_at": now(),
                "zip": ZIP.relative_to(ROOT).as_posix(), "bytes": ZIP.stat().st_size, "sha256": digest(ZIP),
                "members": len(names), "indexed_files": len(index), "model_fits_in_study": 64, "accounts_in_study": 6,
                "builder_new_models": 0, "builder_new_accounts": 0, "security_audit": False,
                "external_review": "NOT_PERFORMED", "upload_performed": False}
    write_json(ZIP.with_suffix(".delivery.json"), delivery, exclusive=True)
    print(json.dumps(delivery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
