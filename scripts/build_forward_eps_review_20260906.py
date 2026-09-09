"""交付前瞻盈利主线、原始预测、财报基础与连续来源的自包含中文审阅包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from scripts.verify_forward_eps_saved_20260906 import verify

DELIVERY=ROOT/"deliverables/510300前瞻EPS研究_20260906"
ZIP=ROOT/"deliverables/510300夏普1.2持续研究_前瞻EPS与金融盈利基础_GPT审阅_20260906.zip"
REPORT="前瞻盈利_估值_股东回报与公募需求.md"
FOLDERS=["510300_original_earnings_source_completion_v1","510300_financial_original_layout_inventory_v1","510300_financial_report_subject_repair_v1",
    "510300_financial_original_facts_v1","510300_financial_original_facts_v1_1","510300_financial_ttm_dependencies_v1",
    "510300_financial_annual_components_v1","510300_financial_ttm_vintage_bridge_v1","510300_financial_quarter_history_v1",
    "510300_forward_eps_source_pilot_v1","510300_forward_eps_source_pilot_v1_1","510300_forward_eps_source_pilot_v2","510300_forward_eps_public_api_probe_v1"]
MANIFESTS=["510300_financial_report_subject_repair_v1","510300_financial_original_facts_v1","510300_financial_original_facts_v1_1","510300_financial_ttm_dependencies_v1",
    "510300_financial_annual_components_v1","510300_financial_ttm_vintage_bridge_v1","510300_financial_quarter_history_v1","510300_forward_eps_source_pilot_v1","510300_forward_eps_source_pilot_v1_1","510300_forward_eps_source_pilot_v2"]


def prepare():
    if (DELIVERY/REPORT).exists():raise FileExistsError("前瞻盈利说明已经准备")
    DELIVERY.mkdir(parents=True,exist_ok=True)
    b=ROOT/"reports/research"
    latest=read(b/"510300_sharpe_1_2_latest_research.json")
    save(DELIVERY/"本轮开始时十轮账户状态.json",latest,exclusive=True)
    pilot=b/"510300_forward_eps_source_pilot_v2"
    eps=pd.read_parquet(pilot/"annual_forecast_vintages.parquet")
    eps=eps.loc[eps.metric_id.eq("ANALYST_ANNUAL_EPS_FORECAST")].sort_values(["report_internal_date","target_fiscal_year"])
    chinese=eps[["sec_name","ts_code","institution","report_internal_date","provider_notice_date","target_fiscal_year","metric_value_exact","eps_definition"]].rename(columns={"sec_name":"公司","ts_code":"证券代码","institution":"预测机构","report_internal_date":"研报落款日","provider_notice_date":"平台发布日期","target_fiscal_year":"预测目标年度","metric_value_exact":"预测每股收益_元","eps_definition":"每股收益口径"})
    chinese.to_csv(DELIVERY/"前瞻每股收益_原始预测十五条.csv",index=False,encoding="utf-8-sig")
    table=["| 研报落款日 | 平台记录的发布日 | 预测目标年度及每股收益（元） |","|---|---|---|"]
    for date,g in eps.groupby("report_internal_date",sort=True):
        text="；".join(f"{r['target_fiscal_year']}年：{r['metric_value_exact']}" for r in g.to_dict("records"))
        table.append(f"| {date} | {str(g.iloc[0]['provider_notice_date'])[:10]} | {text} |")
    factors=(ROOT/"docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md").read_text("utf-8")
    factor_table=factors.split("| 因子 |",1)[1].split("以上是研究定义",1)[0]
    text="""# 510300：前瞻盈利、估值、股东回报与公募需求

**策略研究以未来预期每股收益为主线。完整账户扣除成本后夏普至少1.2的目标仍未达到。** 历史季报、年报和滚动盈利作为预测输入、口径核对和事后检验依据。后续重点是：盈利预期有多高、是否持续上调、市场价格是否已反映这种变化，以及股东回报和资金需求能否提供额外信息。

截至本轮开始，已完成十轮研究，登记161个配置、核算342条完整评价账户，含重复对照；没有全期间达到夏普1.2的账户。本轮完成了前瞻盈利原始资料试验和财报基础建设，没有新增策略账户。这些来源结果不能计为新的成功策略。

## 给管理者看的核心逻辑

对一家公司，可以把价格理解为市场愿意支付的市盈率乘以对每股盈利的判断。用于投资判断时，重点研究未来预期盈利及其变化；计算持有期总回报时，再按实际权益和到账规则计入普通股现金分红。股息率与股价单位不同，不能直接相加；回购已经影响每股收益分母的部分也不能再重复计算一次收益。

公募申购属于资金需求端，会与盈利和估值共同研究。需要同时看赎回，区分净申购与净值上涨，进一步判断资金实际配置到权益资产的比例。资金进入基金，并不意味着同额资金马上买入510300。逆回购则继续汇总各期限，分清投放、到期和净变化。

## 前瞻每股收益的期间和口径

优先研究未来十二个月，同时记录当年、下一年和再下一年的完整年度预测。公开研报主要给年度值，本轮还没有精确的未来十二个月预测。年度值按剩余月份加权只能称为近似值，必须处理季节性和股数变化。

预测上调或下调按**同公司、同机构、同目标年度**比较。跨年后“下一年”代表的年度变了，不能把这个自然变化当成预测上调。市场一致预期还要去除同一机构的重复报告、记录机构覆盖及观点过期情况；公司业绩预告、分析师预测、自建模型预测分别保留来源。

## 每个主要因子的具体中文规则

| 因子 |"""+factor_table+"""
这些是下一轮研究的因子定义；买入、减仓和风险控制规则将在连续数据形成后登记，并实际核算账户。不能仅靠解释听起来合理就认定能获得稳定超额。

## 本轮已经拿到哪些前瞻预测

已保存五份国信证券关于平安银行的原始研报、对应公开页面和日期元数据，提取十五条年度每股收益预测、十五条年度归母利润预测，并构成八组同年度的样本间修正记录。以下为研报原值，均注明“摊薄每股收益，按最新总股本计算”。对象是平安银行，尚不能代表沪深300整体或全市场一致预期。

"""+"\n".join(table)+"""

例如，对2026年的每股收益预测，在已保存的四份研报中依次为2.67、2.47、2.04、2.08元；这能用来识别预期方向变化。中间仍可能有其他研报，不能把八组样本间变化称为完整的三十日或九十日修正因子。[2024年3月原件](https://pdf.dfcfw.com/pdf/H3_AP202403151626825160_1.pdf)，[2024年8月原件](https://pdf.dfcfw.com/pdf/H3_AP202408161639304268_1.pdf)，[2025年3月原件](https://pdf.dfcfw.com/pdf/H3_AP202503161644416306_1.pdf)，[2026年3月原件](https://pdf.dfcfw.com/pdf/H3_AP202603221820687530_1.pdf)。

这些公开原件证明可以找到历史预测资料。网站日期是本次取得的历史元数据，尚未证明它们是当年保存且从未改变的快照；正式历史研究会进一步明确可用时钟和覆盖缺口。本轮未将这些预测接入交易账户。

## 前瞻数据已发现的两个具体问题

**目标年度会错配。** 本次查询2024年3月15日这份研报，公开接口的当前年份为2026，“当年预测每股收益”返回2.67元；原件明确2.24元对应2024年、2.40元对应2025年、2.67元对应2026年。如果把接口的“当年”理解为报告落款年，就会错用盈利预测。后续按原件明确的年度列记录。[该报告公开页面](https://data.eastmoney.com/report/zw_stock.jshtml?infocode=AP202403151626825160)。另查的2022年和2025年研报在此次接口查询中每股收益字段为空，但原件中有明确预测，接口空值不能被填成零。

**落款日与公开日期不同。** 2025年报告落款为3月15日，平台发布记录为3月16日。后续保守时钟采用较晚日期，再匹配交易日；不能把一份后来公开的报告提前一天使用。[对应公开页面](https://data.eastmoney.com/report/zw_stock.jshtml?infocode=AP202503161644416306)。

## 历史财报基础完成到哪里

十五份年报及同期报告新增五十九项核心财务值、四十六项普通股每股收益组成记录。与前一批合并后，共三十九份报告、一百五十三项核心财务值；四条旧广发证券记录仍保留人民币币种未明确状态。两份年报的加权股数换算单位未明确，一项附注为横杠，没有填成数字零。

年报附注使归母利润与普通股利润能够分开：优先股、永续债分配、限制性股票分红可能影响分子；回购库存股、员工持股计划和合并资管产品持股可能影响分母。本轮核对七十五项会计与每股收益关系，十六项年报回归验证和八项版本桥接验证通过。

十二个零散时点的三十六组滚动盈利检查中，三十四组原始数运算齐全，三十一组币种明确；六组后来比较数不同于当年原值，三组年报分季数与次年比較数不同。只有十六组具备明确人民币和一致的年报分季桥接，仍不是完整连续因子。这里没有把三份报告的每股收益机械相加减。

中国人寿2021年前三季度归母利润在三个版本中分别为485.02亿元、484.88亿元和484.86亿元，年报及后续报告说明了同一控制下合并重述。招商证券2023年报告的2022年同期归母比较数比原始2022年季报高191.415599万元，报告说明准则解释第16号递延所得税处理变化。会计版本差异要与经济盈利变化分开。

连续季报原件已完成固定一季报、三季报子集252份中的251份，另1份连接中断尚未取得；216份通过首个文字标题页的初步主体检查，其余已下载报告仍待定位或核对。这个子集暂未提取为新增财务事实；年报、半年报连续序列也尚未齐备。已完成的原件继续供前瞻模型输入和兑现检验使用。

## 接下来的研究顺序

1. 用公开目录扩展历史券商研报，按原件明确年度提取预测；先保留全量索引、缺失和更正记录，再形成覆盖率及同年度修正序列。
2. 按历史成分与明确权重构建沪深300层面的前瞻盈利变化、估值及覆盖指标；盈利预测不足的公司保留缺失，不能只挑当前存续或预测齐全的公司后称为全指数。
3. 在同一有限研究协议内，与价格对照、普通股回报、公募净需求和各期限流动性组合；按当时可用信息滚动训练，完整计入交易成本、分红和空闲现金。
4. 实际检验完整期间及分阶段夏普、超额和回撤，并检查结果是否只依赖少量样本或一个时期。已有十轮失败结果保留；持续研究目标保持开启。

## 本包如何使用

主文和中文明细供直接阅读。原始PDF、公开页面、全文缓存、代码、协议、冻结引用、必要验证及索引在同一包内。只读核对脚本可从解压目录运行，核对保存的来源和数值，不重训模型或重算账户。

本包只对本轮和直接使用的财报基础提供完整复核材料；旧十轮账户以状态背景记录提供，未复制全部历史账户包。未进行安全、隐私或全项目静态审计，未上传或声称外部GPT已经审阅。
"""
    (DELIVERY/REPORT).write_text(text.replace("比較","比较"),encoding="utf-8")
    ar=pd.read_parquet(b/"510300_financial_annual_components_v1/current_original_core_facts.parquet")
    names={"OPERATING_REVENUE_YTD":"营业收入","OPERATING_PROFIT_YTD":"营业利润","PARENT_NET_PROFIT_YTD":"归母净利润","BASIC_EPS_YTD":"基本每股收益"}
    ar["metric_id"]=ar.metric_id.map(names)
    ar[["sec_name","report_period","event_publication_date","metric_id","metric_value_exact","unit","source_page","source_raw_row"]].rename(columns={"sec_name":"公司","report_period":"报告期","event_publication_date":"公告日","metric_id":"指标","metric_value_exact":"精确数值","unit":"数值单位","source_page":"原件物理页","source_raw_row":"原始行"}).to_csv(DELIVERY/"年报及同期报告_五十九项核心值.csv",index=False,encoding="utf-8-sig")
    (DELIVERY/"00_阅读入口.md").write_text("# 阅读入口\n\n先读《"+REPORT+"》，再查看前瞻每股收益十五条CSV和reports/research/510300_forward_eps_source_pilot_v2。财报基础及版本差异在同目录的financial开头文件夹；连续季报已下载251份，主体检查并非全部通过，尚未形成连续因子。\n\n完整原件在data/raw。01_GPT审阅提示.md提供批判性审阅问题；保存结果只读核对.json和FILE_INDEX.csv供核对。脚本不自动连接交易系统。\n",encoding="utf-8")
    (DELIVERY/"01_GPT审阅提示.md").write_text("""# 请审阅前瞻盈利研究主线并给出下一步

用户目标是510300与人民币现金的完整账户扣除成本后夏普至少1.2，并有稳定超额证据；仅使用免费来源。用户明确主要研究前瞻每股收益。十轮161个配置342条评价账户尚未达标，本包没有新增账户。

请重点检查：前瞻年度是否从原件识别，接口相对年份能否用于历史回填；当年、下一年与未来十二个月是否混用；同年度修正是否混入机构变化、年份滚动和股数口径变化；普通股收益与分析师摊薄口径是否混淆；历史元数据的时钟证据是否足够；自建预测与市场一致预期是否混作一种信息。五份研报不是完整序列，请不要用它们推断策略收益。

结合附带的原始财报与差额证据，指出哪些因素能支持前瞻模型，哪些是会计重述或范围变化。请给出可用免费来源、最小连续样本、指数覆盖与权重构造方案，以及优先尝试的有限候选、价格对照、滚动训练、成本与验证规则。区分来源问题、模型问题和账户问题，说明何时应停止某个候选、换新研究设计或仅补充前向观测。

请输出具体批评、下一研究优先级、可执行验证步骤和停止条件。不要把此包的整理、结构检查或正向机制解释当成夏普达标、投资许可或已完成独立验证。
""",encoding="utf-8")
    print("中文前瞻研究说明及明细已生成。",flush=True)


def package():
    if ZIP.exists():raise FileExistsError("前瞻研究交付包已存在")
    if not (DELIVERY/REPORT).exists():raise FileNotFoundError("尚未准备中文说明")
    paths=set()
    for folder in FOLDERS:
        paths.update(p for p in (ROOT/"reports/research"/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for name in MANIFESTS:
        p=ROOT/"config"/(name+"_manifest.json");paths.add(p)
        paths.update(ROOT/r["path"] for r in read(p)["files"])
    paths.update(ROOT/p for p in ["scripts/build_forward_eps_review_20260906.py","scripts/verify_forward_eps_saved_20260906.py","scripts/probe_forward_eps_year_mapping_20260906.py","scripts/retry_financial_quarter_transport_20260906.py","tests/test_financial_annual_components_v1.py","tests/test_financial_ttm_vintage_bridge_v1.py","config/510300_research_authority_v6.json","docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md"])
    # 直接来源的PDF和网页全部复制；不依赖外部ZIP或既有工作区路径。
    for p in list(paths):
        if p.suffix!=".json":continue
        try:d=read(p)
        except (json.JSONDecodeError,UnicodeError):continue
        items=[]
        if isinstance(d,dict):
            if isinstance(d.get("source"),dict):items.append(d["source"])
            items.append(d)
            items+=d.get("rows",[]) if isinstance(d.get("rows"),list) else []
            items+=d.get("all_current_rows",[]) if isinstance(d.get("all_current_rows"),list) else []
        for s in items:
            if not isinstance(s,dict):continue
            for key in ["raw_path","raw_pdf_path","raw_html_path"]:
                if s.get(key) and (ROOT/s[key]).is_file():paths.add(ROOT/s[key])
    for p in sorted(paths):
        if not p.is_file():raise FileNotFoundError(p)
        rel=p.relative_to(ROOT);target=DELIVERY/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    requirements={n:importlib.metadata.version(n) for n in ["pandas","pyarrow","requests","pypdfium2","pdfplumber","pytest"]}
    save(DELIVERY/"运行依赖版本.json",requirements,exclusive=True)
    (DELIVERY/"只读复核说明.md").write_text("# 只读复核\n\n在Python环境安装《运行依赖版本.json》列出的依赖后，从解压目录运行scripts/verify_forward_eps_saved_20260906.py。该脚本默认以自身上级目录为根，核对保存事实、原始行、单位、会计关系、版本差额、前瞻年度、日期元数据和冻结引用。它不重新下载、不训练、不读取策略收益。\n\n源文件收集脚本用于复现来源取得方法，可能涉及网络请求且禁止覆盖完成结果；请优先使用上述只读脚本。旧第八轮缺口队列的上游生成过程以及旧十轮完整账户均不在此包重建范围。\n",encoding="utf-8")
    check=verify(DELIVERY);save(DELIVERY/"保存结果只读核对.json",check,exclusive=True)
    files=sorted(p for p in DELIVERY.rglob("*") if p.is_file() and p.name!="FILE_INDEX.csv")
    index=[{"path":p.relative_to(DELIVERY).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    with (DELIVERY/"FILE_INDEX.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","bytes","sha256"]);w.writeheader();w.writerows(index)
    building=ZIP.with_suffix(".building.zip")
    if building.exists():raise FileExistsError("交付临时包已存在，需要检查后再继续")
    print("独立保存值核对已通过，开始生成与检查完整压缩包。",flush=True)
    with zipfile.ZipFile(building,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DELIVERY.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(DELIVERY).as_posix())
    with zipfile.ZipFile(building) as z:
        assert z.testzip() is None
        members=z.namelist();assert len(members)==len(set(members))
        records=list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert set(members)=={r["path"] for r in records}|{"FILE_INDEX.csv"}
        for r in records:
            content=z.read(r["path"]);assert len(content)==int(r["bytes"]) and hashlib.sha256(content).hexdigest()==r["sha256"],r["path"]
        frozen_refs=0
        for name in MANIFESTS:
            manifest=json.loads(z.read("config/"+name+"_manifest.json"))
            for r in manifest["files"]:
                assert hashlib.sha256(z.read(r["path"])).hexdigest()==r["sha256"],r["path"]
                frozen_refs+=1
    assert building.resolve().parent==ZIP.resolve().parent==(ROOT/"deliverables").resolve()
    building.replace(ZIP)
    receipt={"completed_at":now(),"status":"PASS_SELF_CONTAINED_SOURCE_NUMERICAL_AND_STRUCTURAL_DELIVERY","zip_path":ZIP.relative_to(ROOT).as_posix(),"bytes":ZIP.stat().st_size,"sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        "members":len(members),"indexed_files":len(records),"frozen_references_checked":frozen_refs,"saved_numerical_verification":check,
        "security_audit_performed":False,"uploaded":False,"external_gpt_review_received":False,
        "excluded":"旧十轮完整账户和旧第八轮缺口队列生成上游；提供已冻结直接队列和历史账户状态背景。"}
    save(ZIP.with_suffix(".delivery.json"),receipt,exclusive=True);print(receipt,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--package",action="store_true");a=ap.parse_args()
    prepare() if a.prepare else package()
