"""交付金融盈利口径修复的中文说明、全部原始证据和后续研究入口。"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_original_facts_v1_1 import OUT, REPAIR, norm, get_row, now, write_json
from scripts.verify_financial_original_saved_20260906 import verify

DELIVERY = ROOT / "deliverables/510300金融盈利口径重建_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_金融盈利口径重建_GPT审阅_20260906.zip"
REPORT = "盈利估值股东回报与公募需求_本轮口径修正.md"
FOLDERS = ["510300_original_earnings_source_completion_v1", "510300_financial_original_layout_inventory_v1", "510300_financial_report_subject_repair_v1",
           "510300_financial_original_facts_v1", "510300_financial_original_facts_v1_1", "510300_financial_ttm_dependencies_v1"]
MANIFESTS = ["510300_financial_report_subject_repair_v1_manifest.json", "510300_financial_original_facts_v1_manifest.json",
             "510300_financial_original_facts_v1_1_manifest.json", "510300_financial_ttm_dependencies_v1_manifest.json"]


def read(path):
    return json.loads(path.read_text("utf-8"))


def prepare():
    if (DELIVERY/REPORT).exists():
        raise RuntimeError("金融原始来源交付已经准备")
    DELIVERY.mkdir(parents=True, exist_ok=True)
    result = read(OUT / "result.json")
    facts = pd.read_parquet(OUT / "current_original_financial_facts.parquet")
    eps = read(OUT / "ordinary_share_eps_reconciliation.json")["rows"]
    ttm = read(ROOT / "reports/research/510300_financial_ttm_dependencies_v1/result.json")
    latest = read(ROOT / "reports/research/510300_sharpe_1_2_latest_research.json")
    write_json(DELIVERY / "保存结果只读核对.json", verify(), exclusive=True)
    write_json(DELIVERY / "历史十轮结果背景.json", {"scope": "历史结果背景，本轮不重新训练或重算旧账户", "completed_rounds": latest["completed_rounds"]}, exclusive=True)
    group = read(REPAIR / "page_texts/1218186515.json")
    lines = [{"page": 4, "text": x} for x in norm(group["pages"][3]).splitlines() if x.strip()]
    row = get_row(lines, ["归属于母公司股东的净利润"], [8])
    v = row["values"]
    revision = {"source": group["source"], "raw_summary_row": row,
                "current_ytd_parent_profit_million": str(v[4]), "prior_before_adjustment_shown_in_current_report_million": str(v[5]),
                "prior_after_adjustment_shown_in_current_report_million": str(v[6]),
                "growth_using_before_adjustment_comparison": str(v[4]/v[5]-1),
                "growth_using_after_adjustment_comparison": str(v[4]/v[6]-1),
                "all_three_values_belong_to_2023_report_vintage": True, "earlier_year_original_report_independently_verified_here": False}
    write_json(DELIVERY / "保险会计口径改变导致增长方向不同.json", revision, exclusive=True)
    wide = facts.pivot(index=["ts_code", "sec_name", "report_period", "announcement_id", "event_publication_date"], columns="metric_id", values="metric_value").reset_index()
    wide["report_period"] = pd.to_datetime(wide.report_period, format="mixed")
    wide = wide.sort_values(["ts_code", "report_period"])
    table = ["| 公司 | 报告期 | 原始公告日 | 收入（亿元） | 营业利润（亿元） | 归母利润（亿元） | 基本每股收益（元） |", "|---|---|---|---:|---:|---:|---:|"]
    for r in wide.to_dict("records"):
        op = "摘要未列" if pd.isna(r.get("OPERATING_PROFIT_YTD")) else f"{r['OPERATING_PROFIT_YTD']/1e8:.3f}"
        note = "＊" if r["ts_code"] == "000776.SZ" and r["report_period"].year == 2022 else ""
        table.append(f"| {r['sec_name']}{note} | {r['report_period']:%Y-%m-%d} | {str(r['event_publication_date'])[:10]} | {r['OPERATING_REVENUE_YTD']/1e8:.3f} | {op} | {r['PARENT_NET_PROFIT_YTD']/1e8:.3f} | {r['BASIC_EPS_YTD']:.2f} |")
    report = """# 盈利、估值、股东回报与公募需求：本轮口径修正

**目标仍是完整账户扣除成本后夏普至少1.2，目前没有达到。** 已完成的十轮仍为161个登记配置、342条完整评价账户，含重复对照。本轮补基本面原始数据，没有新增策略回测，不能把来源修复算作策略成功。

## 给老板看的核心说明

研究分成四部分：企业能赚多少钱、市场愿意给多少市盈率、普通股股东实际获得什么回报、公募资金的申购赎回是否形成额外需求。四部分共同用于判断，并按市场状态选择持仓，但能否赚钱仍要接受完整历史账户检验。

“每股盈利乘以市盈率”对应同口径的每股价格。持有期收益还需计入实际有权获得的现金分红。回购影响股数和每股盈利，不能把已反映在每股盈利里的同一影响再次加成收益。公募申购影响供求，可作为解释估值变化的候选信息，不能直接加到企业利润或价格公式里。

只拿当前股价除以市盈率得到每股盈利，再乘回市盈率，不会产生新的预测能力。真正要研究的是：盈利能否持续、市场定价是否偏离可支持的水平、股东回报是否落实，以及资金需求变化是否有独立信息。

## 本轮确实完成了什么

处理最初24份原件及补回的2份集团原件。排除2份错配子公司附件，最终24份正确主体报告取得94项累计财务数值，其中90项明确为人民币口径；另外4项来自一份只写“元”、本表未明确币种的报告，保留为数值诊断。两份建行一季报仅从明确的集团摘要取得3项，缺少营业利润，保持缺失。

第一版只识别82项；后续根据三份真实表格修正版式，增加12项，原有82项的值、页码、单位、列和范围完全保留。并核对了46项会计关系、两组普通股收益扣除关系，未发现剩余数值差额。18项针对真实错误的必要验证通过。主体补回、首次提取、版式修正的记录都在包内。

为了继续形成滚动十二个月盈利，又补了15份年报及上年同期原件，共3863页。初始封面程序确认12份，另外3份在明确的公司信息栏目补充确认：平安公司代码在年报靠后位置，交行为繁体版本，人寿公司全称在靠后页面。这15份已保存全部原文，全年利润、股本和准则可比性尚待提取，不能声称已有12个有效滚动盈利值。

## 问题一：集团公告里可能夹带子公司报告

旧目录按“同一证券、同一季度最早全文报告”保留记录，尚未先排除子公司主体。中国平安两次提前披露的平安银行季报因而被放进集团报告位置。新报告身份必须核对正文中的法定公司全称、证券代码和报告期。

| 报告期 | 错配的附件主体与日期 | 正确集团报告日期 | 本轮处理 |
|---|---|---|---|
| 2023年三季度 | 平安银行，2023年10月25日 | 2023年10月28日 | 更换原件，同步更换公告可用日期 |
| 2024年三季度 | 平安银行，2024年10月19日 | 2024年10月22日 | 更换原件，同步更换公告可用日期 |

两份错误编号在第八轮已保存的有效事实表中均为零条，所以没有证据说这些银行利润已经作为平安集团盈利进入那轮回测；原目录错配确实需要修复。不能由两个案例推断整个公告库都已检查完。[集团2023年原始季报](https://static.cninfo.com.cn/finalpage/2023-10-28/1218186515.PDF)，[集团2024年原始季报](https://static.cninfo.com.cn/finalpage/2024-10-22/1221451424.PDF)。

中信证券两份报告的第一页是无文字图片，第二页明确列示公司全称、代码和季度。它们属于封面提取未识别，不能误报成另外两份主体错配。

## 问题二：归母利润未必全属于普通股股东

平安银行2023年前三季度归母利润396.35亿元，但其中需要扣除优先股股利8.74亿元、永续债利息11.55亿元，普通股可享有部分为376.06亿元。报告列示最新普通股数为194.05918198亿股。

| 计算口径 | 2023年前三季度 | 2024年前三季度 |
|---|---:|---:|
| 归母利润直接除以最新股数 | 2.042418元 | 2.047262元 |
| 扣除其他权益工具回报后的利润除以最新股数 | 1.937862元 | 1.942706元 |
| 原报告用最新股本计算的全面摊薄每股收益 | 1.94元 | 1.94元 |
| 不扣除带来的相对高估 | 5.40% | 5.38% |

这一步核对的是报告明确披露的“最新股本计算值”。基本每股收益原则上使用报告期加权平均普通股股数，不能仅因两位小数相同，就认定最新股数与加权平均股数完全相同。优先股和永续债回报也不能加进普通股股东现金回报。[平安银行2023年原始季报第3至4页](https://static.cninfo.com.cn/finalpage/2023-10-25/1218135621.PDF)，[财政部相关会计规定](https://m.mof.gov.cn/czxw/201403/P020140321350163053512.pdf)。

## 问题三：会计口径改变，可能把增长方向都翻转

中国平安2023年季报明确说明开始执行新保险合同准则，并同时列出2022年调整前和调整后的比较数。该报告列示的2023年前三季度归母利润为875.75亿元；2022年调整前为764.63亿元，调整后为927.81亿元。

使用调整前比较数会算出增长14.53%；使用该报告提供的调整后可比数，则是下降5.61%。这不能归结为市场状态不同，是先要解决的会计可比性问题。上述三个数字都取自2023年这份报告，本轮未把其中2022年比较数冒充2022年当时已经核实的原始报告值。[中国平安2023年原始季报第4页](https://static.cninfo.com.cn/finalpage/2023-10-28/1218186515.PDF)。

中国人寿也有原始版本与后续比较数不同的例子：2021年三季报收入为7277.11亿元、归母利润485.02亿元；2022年三季报提供的2021年重述比较数分别为7277.85亿元和484.86亿元。后续修订只能在其实际披露后使用，原始历史记录保留。[2021年原件](https://static.cninfo.com.cn/finalpage/2021-10-29/1211419194.PDF)，[2022年原件](https://static.cninfo.com.cn/finalpage/2022-10-28/1214937073.PDF)。

## 问题四：合并与母公司、单季与累计、减值前后不能混用

农业银行和交通银行的累计数据在第三列，招商银行在第一列，中国太保2017年报告在第二列，不能统一按第一列或最后一列读取。券商的支出通常以正数列示，本批银行保险多以负数列示；平安银行还需要在减值前利润上扣除信用和其他资产减值。原始表中的母公司利润不作为集团盈利。

## 本批每份报告的数值

下表除建行两份一季报为前三个月累计外，其余均为前九个月累计。金额为了便于阅读换为亿元，精确到分或百万元的原值、单位和页码保存在明细文件中。

""" + "\n".join(table) + """

＊广发证券2022年这份表只标明“元”，本表没有明确人民币。该行按原报告元单位展示换算量级，不作为已确认的人民币总量；四项均保持不可用于人民币因子的状态。

## 四部分因子的具体中文定义和使用边界

| 类别 | 要表达的经济含义 | 计算与使用规则 | 当前进度 |
|---|---|---|---|
| 普通股盈利水平 | 普通股股东能享有多少盈利 | 集团归母利润扣除其他权益工具回报，与同期间普通股股数对应；累计和单季分开 | 本批累计原值已提取，完整滚动盈利及股数仍在补齐 |
| 盈利增长 | 企业赚钱能力是否改善 | 在同主体、同合并范围、同准则、同期间比较；后续重述从后续披露日才生效 | 已证实存在方向翻转实例，需连续原始版本 |
| 盈利质量 | 利润是否由持续业务支撑 | 分行业解释经营利润、减值、投资及公允价值损益；不能把工业企业现金流指标直接套给银行 | 全年附注与行业拆分待提取 |
| 估值水平 | 市场为同口径盈利支付多少价格 | 市值与可归属盈利对应，指数按历史成分及明确权重汇总；不简单平均市盈率、不相加每股收益 | 需补齐完整盈利分母和历史权重后构建 |
| 估值变化 | 定价倍数是否在扩张或收缩 | 盈利变化与价格变化分开，不能把由价格倒推再乘回的恒等式当预测 | 下一研究版本明确独立价格对照 |
| 现金分红 | 普通股持有人有权获得的现金 | 区分分红方案、股权登记、除息及到账；账户内同笔分红只计一次 | 510300历史账户已有真实分红事件规则 |
| 净股份变化 | 回购注销与发行稀释的共同影响 | 分别记录回购计划、实际买入、注销、增发和激励；不把全部回购金额直接当持有人现金收入 | 原始年报股本及执行资料待提取 |
| 单只ETF申赎 | 510300自身的份额需求 | 原始季度申购份额减赎回份额，再除以期初份额；从实际公告后使用 | 已完成第十轮，主方案夏普负0.0135，未达标 |
| 全市场公募需求 | 公募整体对资产配置的需求变化 | 区分股票型、混合型等类别；净资产变化扣除市场涨跌仍需份额及净值时点一致 | 月报分类变更及原始发布日期尚待解决 |
| 央行流动性 | 所有已确认期限的投放与回笼 | 汇总7天、14天、28天等全部期限，买断式及跨月操作分别处理；投放量不能冒充净投放 | 第七轮已用全期限口径检验，未达标 |

公募净申购不能因数值为正就被规定为必然看涨。第十轮最高反向规则夏普0.3114，同样未达到目标。新增基本面研究将保留仅价格、仅盈利估值、加入股东回报、再加入资金需求的对照，比较成本后完整账户结果与跨时期稳定性；具体窗口、权重和持仓规则在下一次收益读取前另行登记。

## 下一步仍然执行的研究

先从已下载的15份原件提取全年利润、上年同期、其他权益工具回报和股份变动，处理准则可比性；然后扩大到连续时期的金融成分样本。满足必要输入后，构建盈利与估值、股东回报、公募需求的有限多因子候选及市场状态组合，并按原有完整账户重跑。当前十轮历史最高仍为事后选择的第六轮月末月初方案，夏普0.5196，不能作为稳定高夏普策略。

本包可从41份PDF、完整文字、冻结版本及保存数值复核本轮事实。历史十轮仅提供已保存的背景结果，不重复打包或重算它们的全部账户；最初缺口目录排名的完整上游第八轮数据亦不重建。已完成必要数值与压缩包结构核对，没有新增安全审计，没有上传或声称取得外部审阅意见。
"""
    (DELIVERY / REPORT).write_text(report, "utf-8")
    (DELIVERY / "00_READ_ME_FIRST.md").write_text("# 阅读导航\n\n目标夏普一点二尚未实现，本包是金融盈利口径修复和下一步原始年报来源，不是第十一轮策略回测。\n\n1. 老板先读《"+REPORT+"》。\n2. 检查《金融原始累计盈利与每股收益.csv》及保存结果只读核对。\n3. 代码、冻结配置、逐份处理记录、41份PDF和全页文字按原路径保存在研究、报告、配置、数据目录。\n4. 将《01_GPT_REVIEW_PROMPT.md》内容连同本包交给审阅者。\n5. 数值复核入口为 scripts/verify_financial_original_saved_20260906.py，解压后从根目录运行该脚本即可；需要 pandas 与 parquet 支持。它只读取保存结果，不请求网络、不重跑回测。\n\n表格提取规则见 docs/510300_FINANCIAL_ORIGINAL_FACTS_V1_1.md。全年及上年同期的15份新原件尚未接纳财务数值；初始封面未识别的3份已用公司信息栏目确认主体，两个层次的记录同时保留。\n", "utf-8")
    (DELIVERY / "01_GPT_REVIEW_PROMPT.md").write_text("# 请审阅本轮真实口径修复，并提出下一步策略\n\n目标是510300与人民币现金完整账户成本后夏普至少1.2，并取得稳定超额证据。十轮161个配置、342条评价账户均未实现目标；本包没有新回测，不要把来源修复当作策略有效。仅用现有免费来源。\n\n请先阅读中文说明，再根据原始PDF、完整文字、逐行原值和冻结版本质疑以下问题：集团与子公司是否混用；合并与母公司、单季与累计、其他权益工具回报与普通股收益是否仍有错配；会计准则和后续重述是否使同比方向失真；用最新股数与加权平均股数能否区分；股东回报与现有账户分红是否重复；全市场公募与单只ETF申赎的经济含义是否混淆。\n\n请指出每个具体问题的原件页码、数值和受影响因子，区分已证实错误、来源尚缺以及策略无效。随后给出下一步研究方向和优先级：如何从新增15份报告建立可靠滚动利润、普通股股数、分红及实际回购变化；如何补连续金融样本；如何构建有限的盈利估值、股东回报、资金需求组合，及状态切换与仅价格对照。给出数据可比条件、避免未来信息的使用时点、完整账户验证、独立验证方式和停止条件。\n\n不要建议事后挑选高夏普区间，也不要把加入因子本身当成产生超额的证据。不得扩大到实盘、券商或订单。\n", "utf-8")
    shutil.copy2(OUT / "金融原始累计盈利与每股收益.csv", DELIVERY / "金融原始累计盈利与每股收益.csv")
    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "pyarrow", "pypdfium2", "pdfplumber", "pytest", "requests"]}
    write_json(DELIVERY / "运行依赖版本.json", versions, exclusive=True)
    (DELIVERY / "必要验证记录.md").write_text("# 必要验证\n\n新版18项真实错误及数值验证通过；此前第一版14项通过，原版82项事实在新版逐项保持一致。另以只读脚本从原始行、金额单位和保存文件复核94项事实、46项会计关系、两组普通股每股收益及41份PDF。精确结果见保存结果只读核对.json。\n\n本轮不重训旧模型、不复算十轮账户；未做额外安全审计。压缩包仅核对可打开、成员唯一、索引覆盖、字节数与哈希。\n", "utf-8")
    print("金融盈利口径修正说明与完整中文数值表已生成。", flush=True)


def dependencies(paths):
    pending = list(paths)
    while pending:
        path = pending.pop()
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text("utf-8-sig"))):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                if name and name.startswith(("research.", "scripts.")):
                    dep = ROOT / (name.replace(".", "/") + ".py")
                    if dep.exists() and dep not in paths:
                        paths.add(dep)
                        pending.append(dep)
    return paths


def package():
    if ZIP.exists():
        raise RuntimeError("金融盈利口径审阅包已存在")
    paths, frozen = set(), {}
    for name in MANIFESTS:
        manifest = ROOT / "config" / name
        paths.add(manifest)
        for rec in read(manifest)["files"]:
            path = ROOT / rec["path"]
            paths.add(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"]
            frozen[rec["path"]] = rec["sha256"]
    for name in FOLDERS:
        paths.update(p for p in (ROOT/"reports/research"/name).rglob("*") if p.is_file())
    for name in ["510300_original_earnings_source_completion_v1", "510300_financial_report_subject_repair_v1", "510300_financial_ttm_dependencies_v1"]:
        paths.update(p for p in (ROOT/"data/raw"/name).rglob("*") if p.is_file())
    for name in ["collect_original_earnings_gaps_20260906.py", "map_financial_original_statement_layouts_20260906.py", "record_annual_subject_section_evidence_20260906.py", "verify_financial_original_saved_20260906.py"]:
        paths.add(ROOT / "scripts" / name)
    paths.add(Path(__file__))
    paths = dependencies(paths)
    for name in ["research", "scripts"]:
        if (ROOT/name/"__init__.py").exists():
            paths.add(ROOT/name/"__init__.py")
    members = {p.relative_to(ROOT).as_posix(): p for p in paths}
    root_files = {p.name: p for p in DELIVERY.iterdir() if p.is_file()}
    members.update(root_files)
    for name, path in members.items():
        if name not in root_files:
            dest = DELIVERY / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    # 使用包内路径复核，确保原始数据和冻结输入齐备。
    receipt = verify(DELIVERY)
    write_json(DELIVERY / "包内路径只读复核.json", receipt, exclusive=True)
    members["包内路径只读复核.json"] = DELIVERY / "包内路径只读复核.json"
    index = []
    temporary = ZIP.with_suffix(".building.zip")
    if temporary.exists():
        raise RuntimeError("存在未完成的临时包，先检查已有进度")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(members.items()):
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if name in frozen:
                assert digest == frozen[name]
            archive.writestr(name, content)
            index.append({"path": name, "bytes": len(content), "sha256": digest})
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(index)
        content = stream.getvalue().encode("utf-8-sig")
        archive.writestr("FILE_INDEX.csv", content)
        (DELIVERY / "FILE_INDEX.csv").write_bytes(content)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and archive.testzip() is None
        assert set(names) == {r["path"] for r in index} | {"FILE_INDEX.csv"}
        for row in index:
            content = archive.read(row["path"])
            assert len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
    assert temporary.stat().st_size < 512000000
    temporary.replace(ZIP)
    receipt = {"completed_at": now(), "zip": str(ZIP), "bytes": ZIP.stat().st_size, "sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
               "members": len(names), "indexed_members": len(index), "frozen_unique_files": len(frozen),
               "crc": "PASS", "unique_members": "PASS", "index_size_hash_coverage": "PASS", "portable_read_only_verification": receipt,
               "original_pdf_count": 41, "new_strategy_evaluations": 0, "security_audit_performed": False, "external_review_received": False, "uploaded": False}
    write_json(ZIP.with_suffix(".delivery.json"), receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融盈利原始口径修复中文交付与审阅包")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--package", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else package()
