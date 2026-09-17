"""生成中文诊断说明和可独立解压复核的原件审阅ZIP。"""
from __future__ import annotations
import argparse
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
sys.path.insert(0,str(ROOT))
from research.eps_disclosed_holdings_diagnostic_v1 import OUT, identity, now, read, save
ZIP = ROOT/"deliverables/510300_EPS实际持仓覆盖诊断_V1_GPT审阅_20260914.zip"


def once(path:Path,value:str) -> None:
    with path.open("x",encoding="utf-8") as stream:
        stream.write(value)


def prepare() -> None:
    result=read(OUT/"result.json")
    closure=read(OUT/"selected_earnings_source_closure.json")
    rows=["| 已公开年报期末 | 诊断原点 | EPS增长覆盖股票市值 | 净利润修正覆盖股票市值 | 修正方向边界 |",
          "|---|---|---:|---:|---:|"]
    for report in result["report_admissions"]:
        growth=next(x for x in result["coverage"] if x["report_year"]==report["year"] and x["field"]=="eps_growth")
        revision=next(x for x in result["coverage"] if x["report_year"]==report["year"] and x["field"]=="profit_revision")
        rows.append(f"| {report['report_end']} | {report['diagnostic_origin']} | {growth['covered_stock_weight']:.2%} | {revision['covered_stock_weight']:.2%} | [{revision['whole_stock_sign_lower_bound']:+.3f}, {revision['whole_stock_sign_upper_bound']:+.3f}] |")
    source=read(OUT/"source_collection_result.json")
    links="\n".join(f"- {x['year']}年度：[上交所完整原报告](https://www.sse.com.cn{x['metadata']['URL']})；目录公告日{x['metadata']['SSEDATE']}。" for x in source['documents'])
    report=f"""# EPS路线诊断：来源可补，现有信息仍不足以确定整体修正方向

只交易510300与现金、成本后夏普至少1.2且复合年化至少10%的目标仍未实现。本轮没有新增策略、收益标签、拟合、账户或随机试验；完成的是旧EPS研究梳理和一个此前未准入的原始基金持仓来源诊断。

## 已经做过的研究及结果

完整账户评价均覆盖2020-01-02至2026-08-14，初始20万元，含现金闲置、整手、费用及分红。双机构共同时钟方法已取得105个有效月，主模型S1压力复合年化2.99%、净夏普0.311、最大回撤-27.58%。估值关系校正Q1压力复合年化3.76%、夏普0.353、最大回撤-32.51%。同窗口买入持有压力年化3.61%、夏普0.287。

原国信EPS控制C0压力年化5.88%、夏普0.609；新增修正分布D1为3.87%、0.407；去掉原PE的Q3为5.60%、0.492。这些均未达到共同目标，不能从多次比较中选择相对好看的一条宣称正确方向。D1、D2、D3早已检验修正分布、价格组合和一致趋势进出场，本轮不把“换成修正广度或加趋势”重新当作新想法。

这些旧数值来自包内完整保存结果、净值收益、成交、年度和阶段记录，本轮不重新模拟旧账户。旧EPS来源校正与六方法对比材料也在包内。新原始持仓来源并未改变任何旧账户或冻结结论。

## 为什么不能只看EPS还在增长

对已有105个公共有效月作事后描述，EPS增长公司中位数105个月全部为正，范围0.116至0.248。这里沿用旧规则的对称增长指标2×(后值-前值)/(后值绝对值+前值绝对值)，不是普通百分增长率或精确NTM。净利润修正中位数则17个月为负、86个月为零、2个月为正。仅用“增长大于零”的二元状态没有负状态，不能承担识别下跌的任务。修正中位数大量为零也缺少状态区分。

这不是对连续数值预测能力的数学否定；连续大小和时序变化仍须由预测与账户证据检验。旧模型已经做过这类检验，结果没有达标。新统计不授权改阈值、调窗口、换聚合器或逆转信号救援。

## 原始基金持仓补上了什么

在计算覆盖前固定2017、2020、2024三个年报。三份报告都从上交所原路径取得；6次请求全部成功。分别提取317、336、337只股票，合计990条完整明细。全部股票公允价值之和与报告股票总额精确到分一致；逐行净值占比均落在两位小数舍入范围。2024年的300只指数投资和37只积极投资均保留。

{links}

年度期末不能代替披露时间。使用目录公告日与PDF送出日较晚者，并等到严格晚于该日的首个已有月末。三个原点距离持仓期末已经89、120、120天。这里的权重只是已披露的旧股票暴露，不能称为当日510300持仓或沪深300实时权重。原历史指数权重阻断保持不变。PDF创建和修改日期均不晚于公告日期，但当前下载与元数据一致仍不构成历史HTTP首发或逐秒可得性证明。

{chr(10).join(rows)}

主分母是原报告所有股票公允价值，另提供基金资产净值口径。基金股票部分分别占净值99.77%、97.16%、98.84%；现金、债券、内部衍生品不被投影为个股盈利暴露。此诊断不是完整基金风险分解。

## 缺失资产能否改变判断

将两机构合并后已知公司的上修记+1、下修记-1、不变记0，乘其旧股票市值权重求和，记为S；未覆盖股票的权重为U。所有未知符号可以在[-1,1]内变化，完整股票的符号和只能确定在[S-U,S+U]。这是缺失值的确定性边界，不是统计置信区间，也不代表全市场分析师共识。

三个预定原点的净利润修正区间全部跨零，未覆盖部分足以改变整体修正方向。2025-04-30的EPS增长符号区间为[+0.447,+0.969]，在这个旧持仓口径下即使未知部分全为负，盈利增长符号仍为正；但其净利润修正区间仍跨零。盈利预计增长、盈利预期上修和随后ETF上涨是不同命题。

在最近一个原点，前十大旧持仓的修正信息缺少美的集团、兴业银行、五粮液。美的和兴业的国信前后报告在保存事实中分别标为“净利润”和“归母净利润”，旧规则不允许默认相等；五粮液的已有证据没有可比较的前次同目标年度记录。对应原件、原始页面、逐机构记录都在包内。这个检查只定位缺口，没有擅自合并口径或重新解析原件。

## 研究方向与停止条件

1. 当前证据不支持继续以“EPS正增长”作为主要涨跌状态，也不支持直接将同一批资料换成市值加权后再跑一轮；三个原点的代表性结果不是新策略收益验证。
2. 如果以后继续补盈利信息，优先级应由未覆盖的重要资产和可比较预测对决定。必须先证明同机构、同公司、同目标年度、同利润归属口径与可用日期，缺失保持未知。没有新增可比较信息时停止；不自动恢复999份、2686份等旧大队列。
3. 美的、兴业两处口径差是可明确核查的局部问题；即使查实可统一，也只能登记新的来源事实版本。不得修改旧版本，不能据此宣称原策略已经修复或按收益反复选择口径。
4. 只有在独立冻结的新信息假设下，先证明来源覆盖与状态区分确实增加，才值得进行新的预测试验。账户需在同窗口、同费用、同标签成熟时钟下对比现有价格控制和仅波动控制，最终仍要求基础和压力成本均满足年化10%、夏普1.2，并经历未参与选型的验证。来源通过本身没有收益含义。
5. 原IF、AH、EPU、解禁公告、EPS分布、估值和其他冻结失败均保留。不能把本轮检查包装为已找到正确投资策略。当前目标状态继续为未达成，本轮属于新增证据进展。

## 交付与验证范围

本包包含三份完整基金年报，以及三个原点直接使用的{closure['source_documents']}份旧券商报告PDF、HTML、原始目录响应、元数据和提取事实；不是只给汇总CSV。也包含冻结代码、配置、990条持仓、1800条机构公司原点记录、旧结果与完整文件索引。

离线脚本从原PDF重新提取所有持仓、封面、资产总额和净值相关页，对比保存文本；从保存EPS事实重算本轮相关字段和公司合并，再验证覆盖与方向边界。旧目录总体完整性、缺少当前原件的所有历史NO_VIEW成因、全部旧EPS的原文解析及旧账户模拟没有在本轮重新执行。659份直接相关原件足以审阅本轮已使用事实，不能被说成涵盖了全历史全部研报。

2024跨行净值标签在诊断冻结前已修正；首次解析预检查的错误保存在回执，没有改动经济规则。包内核对仅说明原件、代码与保存数值之间的一致性，不是外部GPT评审、未来预测验证或实际成交。
"""
    once(OUT/"研究结论与下一步.md",report)
    prompt="""请独立审阅本包，目标仅交易510300和现金，成本后复合年化至少10%、净夏普至少1.2。当前未达标。本轮只有来源与已有信息代表性诊断，没有新策略回测。

先读00_README_FIRST.md、研究结论与下一步.md、冻结协议和result.json，再查原PDF、公司事实和逐只覆盖。请具体批评：
1. 把2017、2020、2024年报的旧持仓延后到首个严格晚于公告日的月末，是否仍有错误的日期或资产含义？有何可证明与不可证明的边界？
2. 股票明细是否完整，净值分母与股票分母是否混淆？2024积极投资、2020证券出借与内部衍生品是否被过度解释？
3. 105个月增长中位数全正、86个月修正中位数为零意味着什么，不能推出什么？是否错误否定了连续信息的预测作用？
4. 三个原点的[S-U,S+U]是否正确，缺失方向可反转的结论是否被错误扩大到未来价格？
5. 原有利润标签冲突和缺少比较记录是否确实定位到原件；怎样用最小、可验收的来源工作区分缺资料与无信息，而不恢复大型队列或调参救援？
6. 结合包内旧EPS账户失败，提出下一步策略研究方向、优先级、免费信息、验证标准和停止条件。不得把已经测试的修正广度、趋势组合、估值变化再命名为新方法；必须明确下一项工作增加了什么新证据。

请给出实质研究质量意见与可执行建议，不要把ZIP、哈希或离线重算通过当作目标达成。本包未上传外部审阅，没有部署或交易授权。
"""
    once(OUT/"GPT审阅提示.md",prompt)
    request="用户原始目标：我们类似的如果没做过可以做一下，最终目标是只操作510300，实现夏普1.2，年化10。\n用户扩展目标：实现夏普1.2，年化10，只操作510300，我们已经做过很多工作了，找到正确的方向，识别下跌，上涨，然后实现我们的目标，你可以加入其他免费有效信息。\n本轮任务：只读梳理已有EPS后，新增三份已披露旧持仓来源的代表性诊断，不新增账户。\n"
    once(OUT/"用户请求与本轮范围.txt",request)
    attachment=Path("E:/CodexData/.codex/attachments/c80a420d-3b55-4771-812c-653b38effab6/pasted-text-1.txt")
    with (OUT/"原始用户附件.txt").open("xb") as target:
        target.write(attachment.read_bytes())
    versions={x:importlib.metadata.version(x) for x in ["numpy","pandas","pyarrow","pdfplumber","pypdfium2","requests","matplotlib"]}
    once(OUT/"requirements-review.txt","\n".join(f"{k}=={v}" for k,v in versions.items())+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout="constrained",gridspec_kw={"height_ratios":[1.1,1]})
    labels=[x["diagnostic_origin"] for x in result["report_admissions"]]
    import numpy as np
    x=np.arange(3)
    for offset,field,label,color in [(-.18,"eps_growth","前瞻EPS增长覆盖","#326f83"),(.18,"profit_revision","同目标年度净利润修正覆盖","#b96540")]:
        vals=[100*next(z["covered_stock_weight"] for z in result["coverage"] if z["origin"]==date and z["field"]==field) for date in labels]
        bars=axes[0].bar(x+offset,vals,width=.33,label=label,color=color)
        axes[0].bar_label(bars,labels=[f"{v:.1f}%" for v in vals],padding=4,fontsize=10)
    axes[0].set_xticks(x,labels)
    axes[0].set_ylim(0,100)
    axes[0].set_ylabel("占原报告股票总市值（%）",fontproperties=font)
    axes[0].legend(prop=font,frameon=False,loc="upper left")
    axes[0].set_title("已有盈利信息覆盖了多少重要资产",fontproperties=font,fontsize=17,loc="left")
    revisions=[next(z for z in result["coverage"] if z["origin"]==date and z["field"]=="profit_revision") for date in labels]
    centers=[z["known_signed_weight"] for z in revisions]
    errors=[z["uncovered_stock_weight"] for z in revisions]
    axes[1].errorbar(centers,x,xerr=errors,fmt="o",capsize=7,color="#b96540",linewidth=3)
    axes[1].axvline(0,color="#67757a",linestyle="--",linewidth=1)
    axes[1].set_xlim(-1.05,1.05)
    axes[1].set_yticks(x,labels)
    axes[1].set_xlabel("旧持仓全股票盈利修正符号的可能区间；不是收益预测",fontproperties=font)
    axes[1].set_title("三个原点均不能排除未覆盖资产反转修正方向",fontproperties=font,fontsize=14,loc="left")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[1] else "y",alpha=.15)
    fig.savefig(OUT/"盈利信息资产覆盖与方向边界.png",dpi=160,facecolor="white")
    plt.close(fig)
    print("中文结论、审阅提示、用户请求和图形已生成。",flush=True)


def package() -> None:
    if ZIP.exists():
        raise FileExistsError("交付包已存在，不能覆盖")
    closure=read(OUT/"selected_earnings_source_closure.json")
    selected={}
    def add(p:Path) -> None:
        if not p.is_file():
            raise FileNotFoundError(p)
        selected[p.relative_to(ROOT).as_posix()]=p
    def tree(p:Path) -> None:
        for path in p.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path)
    tree(OUT)
    raw=ROOT/read(OUT/"source_collection_claim.json")["raw_directory"]
    tree(raw)
    for item in closure["files"]:
        add(ROOT/item["path"])
    paths=["config/510300_eps_disclosed_holdings_diagnostic_v1.json","docs/510300_EPS_DISCLOSED_HOLDINGS_DIAGNOSTIC_V1_PROTOCOL.md",
           "research/eps_disclosed_holdings_diagnostic_v1.py","scripts/collect_510300_eps_disclosed_holdings_v1.py",
           "scripts/prepare_510300_eps_holdings_review_evidence_v1.py","scripts/verify_510300_eps_holdings_packet_v1.py",
           "scripts/extract_510300_eps_holdings_pdf_pages_v1.py",
           "scripts/build_510300_eps_holdings_review_20260914.py",
           "research/forward_eps_two_institution_features_v3.py","research/forward_eps_monthly_policy_v1.py",
           "research/forward_eps_coverage_representativeness_v1.py","docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md"]
    for rel in paths:
        add(ROOT/rel)
    tree(ROOT/"reports/research/510300_forward_eps_two_institution_features_v3")
    support=read(OUT/"saved_eps_support_diagnostic.json")
    for item in support["existing_account_source_files"]:
        add(ROOT/item["path"])
    for name in ["510300_forward_eps_two_institution_policy_v3","510300_forward_eps_valuation_consistency_policy_v2","510300_forward_eps_revision_distribution_policy_v1"]:
        tree(ROOT/"reports/research"/name)
        add(ROOT/"config"/(name+".json"))
    comparison=read(ROOT/"reports/research/510300_forward_eps_source_v4_replay_comparison_v1/result.json")
    for item in comparison["files"]:
        add(ROOT/item["path"])
    for name in ["510300_forward_eps_csi_facts_v2","510300_forward_eps_soochow_facts_v4"]:
        for filename in ["result.json","saved_source_verification.json"]:
            p=ROOT/"reports/research"/name/filename
            if p.exists():
                add(p)
    verified=ROOT/"reports/research/510300_eps_holdings_delivery_verification_20260914/workspace_verification.json"
    if read(verified)["status"]!="PASS_OFFLINE_ORIGINAL_TABLE_EPS_FACT_AND_COVERAGE_RECOMPUTATION":
        raise ValueError("工作区离线重算未通过")
    add(verified)
    readme=r"""# 510300 EPS实际持仓覆盖诊断V1

先读 reports/research/510300_eps_disclosed_holdings_diagnostic_v1/研究结论与下一步.md，再读同目录GPT审阅提示.md。当前目标仍未达成；本轮0新策略、0新收益标签、0拟合、0账户。三个旧持仓报告与已有盈利事实揭示重要资产覆盖缺口。

本包包含三个基金原始完整年报及659份直接相关券商原件、原始HTML和目录响应，990条股票持仓和1800条机构公司原点证据；包含旧EPS账户及训练保存物供交叉审阅。旧EPS目录全集的完整性和所有缺失分类没有在本轮重新证明。这里有历史日期的旧基金持仓，不是当日指数权重；三个诊断原点不能外推全历史或实时状态。

Windows复核：把ZIP完整解压到任意新目录，在该目录打开PowerShell。使用已有包含numpy、pandas、pyarrow、pdfplumber的Python；所用版本见研究目录requirements-review.txt。运行 python .\scripts\verify_510300_eps_holdings_packet_v1.py --receipt .\离线复核回执.json。该脚本不联网、不拟合、不生成账户，先验证FILE_INDEX，再从基金PDF提取相关页面，重算保存盈利事实与覆盖边界。不要运行采集器或旧回测入口来“验证”本包。

FILE_INDEX.csv覆盖除索引自身以外全部ZIP成员，提供相对路径、字节数与SHA-256。ZIP检查和数值复核不等于外部研究评审或交易授权。用户请求和原始附件已在研究目录保存。
"""
    memory={"00_README_FIRST.md":readme.encode("utf-8")}
    files=[]
    for rel,p in sorted(selected.items()):
        data=p.read_bytes()
        files.append({"path":rel,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    files += [{"path":rel,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()} for rel,data in memory.items()]
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=["path","size_bytes","sha256"])
    writer.writeheader();writer.writerows(sorted(files,key=lambda x:x["path"]))
    temporary=ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,p in sorted(selected.items()):
            z.write(p,rel)
        for rel,data in memory.items():
            z.writestr(rel,data)
        z.writestr("FILE_INDEX.csv",buffer.getvalue().encode("utf-8-sig"))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):
            raise ValueError("ZIP CRC或重名检查失败")
        expected={x["path"]:x for x in files}
        if set(z.namelist())!=set(expected)|{"FILE_INDEX.csv"}:
            raise ValueError("ZIP成员与索引不等")
        for rel,row in expected.items():
            data=z.read(rel)
            if len(data)!=row["size_bytes"] or hashlib.sha256(data).hexdigest()!=row["sha256"]:
                raise ValueError("ZIP文件与索引哈希不匹配")
    if temporary.stat().st_size>=512_000_000:
        raise ValueError("ZIP超出512MB上限，保留building对象，不静默丢弃原件")
    os.replace(temporary,ZIP)
    receipt={"status":"PASS_REVIEW_ZIP_INDEX_CRC_AND_MEMBER_HASHES", "created_at":now(),"path":str(ZIP),
             "bytes":ZIP.stat().st_size,"sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
             "members":len(files)+1,"indexed_members":len(files),"uncompressed_bytes":sum(x["size_bytes"] for x in files),
             "new_fund_original_pdfs":3,"reused_earnings_original_pdfs":closure["source_documents"],
             "new_models":0,"new_accounts":0,"external_review_received":False,"security_audit":False,"goal_achieved":False}
    save(ZIP.with_suffix(".delivery.json"),receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["prepare","package"],required=True)
    args=parser.parse_args()
    prepare() if args.mode=="prepare" else package()
