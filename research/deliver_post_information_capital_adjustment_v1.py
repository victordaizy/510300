"""整理研究结论、直接数值证据和可离线复算的审阅包。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber
from pdfplumber.utils import extract_text

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_post_information_capital_adjustment_v1"
PRIOR = ROOT / "reports/research/510300_expectations_policy_evidence_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare() -> None:
    events = pd.read_csv(STUDY / "results/六次事件_观察后五日完整结果.csv")
    summary = json.loads((STUDY / "results/summary.json").read_text(encoding="utf-8"))
    forecast_rows = pd.read_csv(PRIOR / "inputs/wgc_all_extracted_forecasts.csv")
    relevant = forecast_rows[forecast_rows.source_sha256.isin(events.source_sha256)].drop_duplicates(["source_sha256", "series"])
    checks = []
    for h, group in relevant.groupby("source_sha256"):
        record = group.iloc[0]
        source = ROOT / record.source_path
        assert sha(source) == h
        with pdfplumber.open(source) as pdf:
            page = pdf.pages[int(record.page)-1]
            # 接近零的矩阵误差不能把实际横排数字误认成竖排。
            chars = []
            for c in page.chars:
                if c["x0"] > page.width * .51:
                    continue
                a, b, cc, d, _, _ = c["matrix"]
                chars.append(dict(c, upright=True) if a > 0 and d > 0 and abs(b) < 1e-5 and abs(cc) < 1e-5 else c)
            text = extract_text(chars, x_tolerance=2, y_tolerance=2)
            for row in group.itertuples():
                match = re.search(r"CN\s+Money\s+Supply\s+" + row.series + r"\s+YoY\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", text, flags=re.M)
                assert match, (row.report_date, row.series)
                previous, forecast = float(match.group(1)), float(match.group(2))
                assert np.isclose(previous, row.previous_pp) and np.isclose(forecast, row.forecast_pp)
                checks.append({"report_date": row.report_date, "series": row.series,
                               "previous_pp": previous, "forecast_pp": forecast, "page": int(row.page),
                               "source_url": row.source_url, "article_url": row.article_url,
                               "source_sha256": h, "limited_numeric_row": match.group(0).strip(),
                               "source_local_hash_and_numbers_checked": True})
    assert len(checks) == 12
    js(STUDY / "sources/六份预期报告_12行数字核对.json", checks)
    for name in ("510300_货币预期与政策全景", "510300_M1M2_完整历史走势"):
        for ext in ("png", "svg"):
            shutil.copy2(PRIOR / "figures" / f"{name}.{ext}", STUDY / "figures" / f"{name}.{ext}")
    for name in ("post_information_capital_adjustment_v1.py", "verify_post_information_capital_adjustment_v1.py", Path(__file__).name):
        shutil.copy2(ROOT / "research" / name, STUDY / "code" / name)
    shutil.copy2(PRIOR / "source_records/input_origins.json", STUDY / "evidence/prior_input_origins.json")
    shutil.copy2(PRIOR / "source_records/policy_context_source_receipts.json", STUDY / "evidence/policy_context_source_receipts.json")
    # 原图的全部日频数据直接导出，不重新采样或另选历史区间。
    pd.read_parquet(STUDY / "inputs/money_market_daily.parquet").to_csv(STUDY / "inputs/money_market_daily.csv", index=False, encoding="utf-8-sig")
    pd.read_parquet(STUDY / "inputs/market_daily.parquet").to_csv(STUDY / "inputs/market_daily.csv", index=False, encoding="utf-8-sig")
    sources = [
        {"title": "央行104个月金融统计", "support": "M1/M2实际值、公布时刻、口径；逐月链接见输入CSV，原始HTML在sources/official_money。", "checked": "LOCAL_SOURCE_HASH_MATCH"},
        {"title": "世界黄金协会六份Weekly Markets Monitor", "url": "https://www.gold.org/goldhub/gold-focus/2025/07/weekly-markets-monitor-big-data-little-reaction", "support": "报告日期及Bloomberg共识的有限数字摘录；不等于公布前最后一刻共识。", "checked": "ARTICLE_LIVE_AND_SIX_SAVED_PDF_HASHES_12_VALUES"},
        {"title": "上交所ETF常见问题", "url": "https://etf.sse.com.cn/fund/quertion/", "support": "股票ETF T+1、100份一手、实物申购通过股票篮子创造份额；开市前公布PCF不等于公布真实净申赎或份额规模时间。", "checked": "LIVE_PRIMARY_SOURCE"},
        {"title": "Jarocinski与Karadi原文摘要", "url": "https://www.aeaweb.org/articles?id=10.1257/mac.20180090", "support": "公告同时传递政策与经济信息；联合市场反应可协助识别。不是510300已有效的证据。", "checked": "LIVE_PRIMARY_SOURCE"},
        {"title": "Shieh作者政策因子说明", "url": "https://harrisonshieh.com/research/", "support": "作者披露当前因子全样本PCA；需要截断样本版本。本轮不把它用于历史可交易信号。", "checked": "LIVE_PRIMARY_SOURCE"},
        {"title": "McLean与Pontiff", "url": "https://onlinelibrary.wiley.com/doi/10.1111/jofi.12365", "support": "97个横截面预测变量的样本外/发表后衰减，不证明本仓库全部失败源于机构竞争。", "checked": "LIVE_PRIMARY_SOURCE"},
        {"title": "Bailey与Lopez de Prado作者论文", "url": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf", "support": "DSR考虑选择偏差及非正态；不能以最后四组代替完整试验历史。", "checked": "LIVE_AUTHOR_COPY"},
        {"title": "Gabaix与Koijen", "url": "https://www.nber.org/papers/w28967", "support": "附件机制线索；本轮原网页403，PDF/作者替代地址也未成功，不据此增加量化结论。", "checked": "ACCESS_FAILED_NO_NEW_CLAIM"},
    ]
    js(STUDY / "sources/来源与本轮核查范围.json", sources)
    rows = []
    for r in events.itertuples():
        rows.append(f"| {r.stat_month} | {r.spread_surprise_proxy_pp:+.1f} | {r.initial_response:+.2%} | {r.entry_date}—{r.exit_date} | {r.residual_5d_gross_return:+.2%} |")
    report = "\n".join([
        "# 510300：消息之后是否还有可参与的五日行情", "",
        "**结论：当前证据不支持把‘M1/M2剪刀差超预期就买、低于预期就卖’直接作为交易规则。六次可核实事前预期案例中，首轮反应5次同向，但观察结束、次日开盘后的固定五日只有2次同向。信息出现后的持续资金调整仍是未完成验证的假设；资金时点和样本数量不足，四组模型、完整账户及回撤退出比较均未运行。目标夏普1.2、年化10%尚未实现。**", "",
        "用户已确认20万元为主账户本金，2万元仅作成本对照。行情冻结到2026-09-11；覆盖2012-05-28起3476个交易日。货币数据覆盖2018-01至2026-08共104个月。2026-09-21是本次复核日期，不代表行情或宏观数据已更新至当日。", "",
        "## 1. 此次新增的实证结果", "",
        "主时钟在新收益计算前冻结：公告后第一个完整交易日收盘观察，下一交易日开盘开始，第五交易日预定收盘结束。晚间或节假日公布时，第一个完整交易日会顺延。2026年1月数据在春节前公布，观察日为2月24日，入场为2月25日，不能把假期计作持有交易日。", "",
        "| 数据所属月 | 剪刀差预期差（百分点） | 首轮反应 | 新五日区间 | 可参与毛收益 |", "|---|---:|---:|---|---:|", *rows, "",
        "四次正预期差中，延迟五日两次上涨、两次下跌；两次负预期差，延迟五日均上涨。2025年6月数据是明显延续的案例；2026年1月数据先涨后回落，2026年6月数据先跌后反弹。全部案例一起看，不能固定解释成同一条延续规律。", "",
        "这里的2/6是六个历史案例的描述，不是模型预测准确率，也不构成‘应该反向交易’的证据。没有拟合参数、没有显著性通过，也没有研究这些事件在相同趋势、风险和政策条件下的因果增量。", "",
        "![前后口径比较](figures/公告后反应与延迟五日收益.png)", "",
        "两个五日窗口的起止日不同，柱状图不能相减解释为被首轮反应消耗的收益。结果CSV另外提供同一新退出日的严格拆分：公告前收盘到实际入场开盘，以及入场开盘到退出，二者按复利相乘还原完整收益。六个窗口均没有分红；通用账户仍必须区分登记、除息和到账日期。", "",
        "![完整路径](figures/六次事件_初始反应与可参与路径.png)", "",
        "## 2. 预期、政策和资金各能说明什么", "",
        "剪刀差定义为M1同比减M2同比；预期差为实际剪刀差减事前预期剪刀差，均用百分点。M1与M2各自超预期的部分也保留，避免把M1加速与M2减速解释成同一经济机制。2025年M1口径变更前后不拼接阈值。", "",
        "104个月均保留在准入清单中：6个月有符合本轮历史案例条件的配对预期，98个月没有。六份预期来自公布前4—9天的周报，是分别汇总的M1/M2调查数字之差，不是同一批受访机构逐一计算后的剪刀差中位数，也不是公布前最后一刻的市场预期。六份原PDF的哈希和12个关键数字已经重新核对；这仍不能证明历史不可变版本。", "",
        "![走势及政策背景](figures/510300_货币预期与政策全景.png)", "",
        "政策确实应纳入解释，但本轮八项政策只用于图示背景，清单并不完整，缺少对应的事前预期，不能给每一段上涨分配政策贡献。此前446条政策记录另存，其中部分在ETF上市前；作者的Target/Path因子由全样本PCA构造，当前版本没有准入逐期预测。需要公告原文、首次时间、宣布与实施的区别、当时可知的预期及其他同时发生的信息。", "",
        "政策消息也可能同时释放经济前景信息，所以‘宽松’不能机械映射为上涨；参考[AEA原论文](https://www.aeaweb.org/articles?id=10.1257/mac.20180090)。作者[政策因子说明](https://harrisonshieh.com/research/)明确提示全样本PCA限制。", "",
        "现有日份额资料的直接回执已于9月14日完成局部核查：1216行数值及2220条较早历史响应没有建立逐日首次公布时间。此次读取并保留该结果，没有将保存时间改成公布时间。上交所开市前公布的是申购赎回清单，不能据此推断每个历史统计日的真实份额净变化当时已知；份额增加也不能直接证明方向性资金意图。参考[上交所说明](https://etf.sse.com.cn/fund/quertion/)。", "",
        "## 3. 四组比较与停止状态", "",
        "| 组别 | 冻结输入 | 现状 |", "|---|---|---|",
        "| A | 公告前20日变化、首轮反应、已知20日波动 | NOT_RUN：共同事件样本不足 |",
        "| B | A加一个剪刀差预期差 | NOT_RUN：共同事件样本不足 |",
        "| C | A加一个有历史公开时间证据的份额变量 | NOT_RUN：样本与资金时点均未通过 |",
        "| D | A加消息、份额及唯一消息乘份额交互项 | NOT_RUN：样本与资金时点均未通过 |", "",
        "协议事前固定至少36个训练事件群、24个评价事件群；这是最低执行条件，不是统计功效已经足够的保证。六例无法达到，2025年新口径总月数本身也少，不是接下来补几条就能解决。2025年前旧口径即便取得更多预期也必须单独研究，不能无条件拼接。", "",
        "波动用于风险度量和预算，不能从本轮六例推出风险预测已改善。支撑位只记录公告前20日最低价及以当时已知波动标准化的距离，不以后来谷底倒画。资金压力回归保留为另一机制，本轮不根据两个负消息反弹临时改成反向策略。", "",
        "预先登记的一项回撤退出为：上涨达到入场前已知日波动的2倍后启动，收盘从持有期收盘高点回撤达到1倍日波动，次日开盘卖出，最迟仍第五日预定收盘。该阈值是待检验设定，没有最优性证据；入场门未通过，当前没有运行退出比较。不会把未来最高价、盘中止损线或触发当天不可卖份额当作成交。", "",
        "## 4. 20万元与2万元的费用区别", "",
        "按原研究成本设定，佣金每边万二、最低5元，基础/压力滑点分别每边5/10bp。以参考价始终4元的整手往返为例，20万元满预算约需0.140%/0.240%的价格上涨才能覆盖基础/压力费用；2万元满预算约需0.151%/0.251%；2万元仅使用25%预算时约需0.309%/0.409%。完整8组示例已保存。", "",
        "这些是固定价格的费用算例，没有交易信号，也没有生成净值账户。若未来入场增量通过门槛，再在同一引擎内计入整手、T+1、跳空、分红、闲置现金和全部交易日，并比较BUY_HOLD及相近仓位/风险基准。当前收益、净夏普及回撤的账户结果保持未计算，不能填零或以六笔交易年化代替。", "",
        "## 5. 对附件事实的修正", "",
        "附件提及的24候选相关系数中位数0.5221、协方差参与率2.41，以及27区间前后排名均值−0.0398，均与保存材料一致。此次从24×24相关矩阵和27行排名记录复算了前者相关中位数与后者排名均值；参与率仅核对已有结果，没有重新跑24个账户。它不等于已证明只有2.41个独立经济机制。", "",
        "**附件关于NBS仍停在数据门的说法已经过时。**2026-09-05的限定五分钟用途获批后，88个合格事件、86个模型事件完成首次G2；50个评价事件的增强模型MSE比基准高7.82%，三个时代均未改善。因此该家族已经冻结拒绝，G3/G4未运行。本轮没有重启该家族。旧85/15策略仍保持用户终止状态。", "",
        "DSR需要完整选择历史及实际账户收益分布；本轮没有新账户，且上游试验分布并不完整，所以不计算一个外表精确的校正夏普。参考[作者论文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。", "",
        "## 6. 最终取舍", "",
        "当前保留M1/M2、预期差与政策作为市场背景和事件观察资料；不升级为自动方向规则，不认定持续资金调整增量已成立。‘上涨后回撤退出’可以保持预先定义，但它必须依附于先通过验证的入场，当前没有依据认为它能补救信息不足。", "",
        "继续推进所需的外部新证据是：足量同口径、带公布前版本时间的配对共识；可还原历史公开时点的真实资金变量；若使用政策价格因子，则须原始构造数据和逐期估计。达到这些条件前，重画曲线、扩大价格指标或调整窗口都不算信息补齐。本轮已完成能复核的六例执行时钟研究及资料核对，完整四组因果/预测比较和账户目标仍未完成。", "",
        "所有图、每日数据、事件清单、冻结协议、代码及失败状态均在审阅包中。数值复核通过不代表外部GPT已审阅，也不代表建立独立验证。",
    ])
    (STUDY / "研究结论.md").write_text(report + "\n", encoding="utf-8")
    requirements = [
        ("补齐走势图、预期差和政策背景", "COMPLETED_WITH_EXPLICIT_GAPS", "4类PNG/SVG图，3476日行情及104月清单；缺失预期不补造"),
        ("去除已发生初始反应，固定后续五日", "COMPLETED", "6例全部交易时钟、48路径点及42项独立数值核对"),
        ("核实附件引用的已有仓库结论", "COMPLETED", "相关/排名复核；纠正NBS过时状态"),
        ("四组增量模型", "NOT_RUN_DATA_GATES", "6例不足36训练+24评价；C/D另缺资金时点"),
        ("资金持续调整机制", "NOT_ESTABLISHED", "现有历史份额资料不能证明决策时已公开"),
        ("政策增量及竞争解释", "NOT_ESTABLISHED", "背景事件不完整；全样本PCA不准入；不能做因果归因"),
        ("单一回撤退出与固定退出比较", "NOT_RUN_ENTRY_GATE", "协议已登记；入场未成立，不生成比较账户"),
        ("20万主本金/2万对照", "CAPITAL_CONFIRMED_COST_EXAMPLES_COMPLETE", "8个整手费用例；非策略账户"),
        ("全账户夏普1.2与年化10%", "NOT_ACHIEVED_NOT_RUN", "全部现金日、费用、风险匹配及DSR留待合格账户"),
        ("独立前向验证", "NOT_ESTABLISHED", "0新前向事件；旧历史不能改名"),
    ]
    pd.DataFrame(requirements, columns=["requirement", "status", "evidence_or_gap"]).to_csv(STUDY / "需求完成与缺口.csv", index=False, encoding="utf-8-sig")
    js(STUDY / "status.json", dict(summary, completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                  research_scope_complete="本轮病例执行时钟及来源核对已完成；完整资金增量目标仍受外部数据和样本限制",
                                  forward_collection_scheduled=False, main_capital_cny=200000, comparison_capital_cny=20000))
    (STUDY / "00_README_FIRST.md").write_text("""# 阅读顺序

先看《研究结论.md》和figures下四组图，再看《需求完成与缺口.csv》、protocol.json、results及verification.json。

本包包含独立复算所需完整日行情、104月正式发布及原始HTML、六例信息/新旧收益、政策背景及原政策事件表、历史裁决、冻结文件清单和代码。六份周报仅保留必要数字摘录、公开链接、页码及原PDF哈希，不分发全文。资金2220条原始响应未重复装入；该项仅据直接关联的既有来源裁决判断缺口，不宣称重新核查全部响应。

可在已安装依赖的Python环境中从解压目录运行code/post_information_capital_adjustment_v1.py，参数--study-dir为解压目录、--output-dir为新的复算目录；验证命令使用code/verify_post_information_capital_adjustment_v1.py --study-dir 解压目录。绘图加--plot。不得对已有冻结结果目录运行建库脚本。冻结/打包脚本用于原仓库环境，离线数值复算与核对脚本不依赖原仓库。

模型、账户和退出比较均为NOT_RUN，代码没有把这些未运行阶段伪装成已实现的交易系统。当前没有订单、外部上传或自动收集任务。
""", encoding="utf-8")
    (STUDY / "用户需求与本轮口径.md").write_text("用户要求510300与M1/M2走势对比，加入预期差及宏观政策，结合大幅上涨后的回撤退出，补齐信息并得出结论。后续附件要求先检验观察初始反应后的五日信息/资金增量，资金不准入时C/D不运行。2026-09-21用户确认：20万元为主，2万元作成本对照。不得把未完成的数据与账户目标写为完成。\n", encoding="utf-8")
    (STUDY / "GPT审阅提示词.md").write_text("""请审阅这个510300研究包。先读研究结论、需求完成与缺口、protocol、freeze_receipt及status，复核六例观察日/次日开盘/第五日收盘时钟和收益拆分，检查是否仍有前视、日期错位或同一事件多算。

本轮四组模型及账户未运行。请区分：历史案例描述、来源证据不足、信息增量未建立、已冻结失败；不要把2/6当预测胜率，也不要把首轮反应5/6当可交易成绩。政策八事件是背景而非完整抽样，公开政策因子全样本PCA不合格。六份周报预期并非即时共识，只有数值摘录与链接，外部原文仍可核查。

请指出最高优先级的具体错误、现有证据支持/不支持的结论，并提出下一项有独立经济机制的策略方向、所需新数据、预先验证条件和停止条件。评价月频新口径样本约束是否使该方向近期不适合作为主要alpha来源。不要建议改历史窗口、救援NBS或重启已终止85/15。主账户20万元、成本对照2万元；最终门槛仍净夏普1.2且复合年化10%，现金日不能省略。审阅包不是订单或实盘授权，未声称已获外部审阅。
""", encoding="utf-8")
    (STUDY / "排除清单.md").write_text("""# 包内边界

- 未装入六份WGC报告全文，只装必要数值摘录、页码、URL及已核对的PDF哈希。
- 未装入2220条旧份额原始响应和1216行第三方接口原始缓存；仅保存已有完整来源裁决。本轮不从缺少发布时间的数据构造资金模型。
- 未装入24个旧策略的全部账户；相关矩阵和27区间排名可独立复算，协方差参与率仅引旧结果。
- 没有本轮四组模型、净值、退出对照或独立验证数据，状态明确NOT_RUN/NOT_ESTABLISHED。
- 旧政策图只提供背景；八事件不是完整政策类别样本。
- 研究主数据止于2026-09-11，未伪装为9月21日行情。
""", encoding="utf-8")
    versions = "\n".join(f"{x}=={importlib.metadata.version(x)}" for x in ("numpy", "pandas", "pyarrow", "matplotlib", "pdfplumber"))
    (STUDY / "requirements.txt").write_text(versions + "\n", encoding="utf-8")
    print("结论、来源数字核对及交付资料已整理。")


def package() -> None:
    deliver = ROOT / "deliverables"
    final = deliver / "510300_消息后剩余行情与资金增量_V1_GPT审阅_20260921.zip"
    temporary = final.with_suffix(".building.zip")
    if final.exists() or temporary.exists():
        raise RuntimeError("交付文件已存在，禁止覆盖。")
    files = sorted(p for p in STUDY.rglob("*") if p.is_file() and p.name != "FILE_INDEX.csv")
    index = [{"path": p.relative_to(STUDY).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    with (STUDY / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files + [STUDY / "FILE_INDEX.csv"]:
            archive.write(file, file.relative_to(STUDY).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names)) == len(index) + 1
        for row in index:
            raw = archive.read(row["path"])
            assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
    temporary.replace(final)
    extracted = deliver / "510300_消息后剩余行情_V1_解压复核_20260921"
    if extracted.exists():
        raise RuntimeError("解压验证目录已存在。")
    with zipfile.ZipFile(final) as archive:
        archive.extractall(extracted)
    command = [sys.executable, str(extracted / "code/verify_post_information_capital_adjustment_v1.py"), "--study-dir", str(extracted)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    fresh_verification = json.loads(process.stdout)
    recomputed = deliver / "510300_消息后剩余行情_V1_全量复算_20260921"
    command = [sys.executable, str(extracted / "code/post_information_capital_adjustment_v1.py"), "--study-dir", str(extracted), "--output-dir", str(recomputed)]
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    compared = []
    for old in sorted((extracted / "results").glob("*.csv")):
        pd.testing.assert_frame_equal(pd.read_csv(old), pd.read_csv(recomputed / old.name), rtol=1e-11, atol=1e-12)
        compared.append(old.name)
    assert json.loads((extracted / "results/summary.json").read_text(encoding="utf-8")) == json.loads((recomputed / "summary.json").read_text(encoding="utf-8"))
    receipt = {"status": "PASS_ZIP_STRUCTURE_FRESH_EXTRACTION_AND_SAVED_CASE_RECOMPUTATION", "archive": str(final),
               "bytes": final.stat().st_size, "sha256": sha(final), "members": len(index) + 1, "indexed_members": len(index),
               "csvs_recomputed": compared, "fresh_extraction_verification": fresh_verification,
               "new_model_fits": 0, "new_account_runs": 0, "external_review": "NOT_PERFORMED", "account_target_achieved": False}
    js(deliver / "510300_消息后剩余行情_V1_交付回执_20260921.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "package"):
        raise SystemExit("参数应为prepare或package。")
    prepare() if sys.argv[1] == "prepare" else package()
