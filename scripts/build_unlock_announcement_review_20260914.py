"""打包本轮完整公告来源、失败证据、冻结模型与保存账户。"""
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

OUT = ROOT/"reports/research/510300_unlock_announcement_increment_v1"
SOURCE = ROOT/"reports/research/510300_unlock_announcement_source_admission_v1"
VERIFY = ROOT/"reports/research/510300_unlock_delivery_verification_20260914"
ZIP = ROOT/"deliverables/510300_解禁公告信息增量_V1_GPT审阅_20260914.zip"
NAMES = {"M0":"价格与季节基线", "M1":"加入解禁公告密度", "VOL_ONLY":"仅波动控制", "BUY_HOLD":"买入持有"}


def once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(text)


def plot_saved():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout="constrained",gridspec_kw={"height_ratios":[1.6,1,1]})
    colors={"M0":"#597b95","M1":"#bd523c","VOL_ONLY":"#2e826b","BUY_HOLD":"#85878b"}
    for model,label in NAMES.items():
        d=pd.read_parquet(OUT/"accounts"/("STRESS_"+model+"_ledger.parquet"))
        equity=d.equity/200000*100
        axes[0].plot(d.date,equity,label=label,color=colors[model],linewidth=1.5)
        if model in ("M1","VOL_ONLY"):
            axes[1].plot(d.date,(d.equity/d.equity.cummax()-1)*100,label=label,color=colors[model],linewidth=1.2)
    features=pd.read_parquet(OUT/"features.parquet")
    features=features.loc[features.date.between("2020-01-02","2026-08-11")]
    axes[2].step(features.date,features.ANN20_MEMBER_COUNT,where="post",color="#6d598b",linewidth=1.0)
    axes[0].set_title("解禁公告密度：未建立可靠的510300择时优势",fontproperties=font,fontsize=17,loc="left")
    for ax,label in zip(axes,["压力成本净值（初始=100）","区间内最大值回撤（%）","20个原点内有公告的成员数"]):
        ax.set_ylabel(label,fontproperties=font,fontsize=10)
        ax.grid(alpha=.17)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(prop=font,ncol=2,frameon=False,loc="upper left")
    axes[1].legend(prop=font,ncol=2,frameon=False,loc="lower left")
    axes[2].set_xlabel("2020-01-02至2026-08-11；固定历史试验，非独立未来验证",fontproperties=font,fontsize=10)
    fig.savefig(OUT/"解禁公告压力账户与信息.png",dpi=160,facecolor="white")
    plt.close(fig)


def write_reports(result):
    table=["| 账户 | 费用 | 复合年化 | 净夏普 | 最大回撤 | 平均持有比例 |","|---|---|---:|---:|---:|---:|"]
    for row in result["accounts"]:
        table.append(f"| {NAMES[row['model']]} | {row['cost']} | {row['annualized_return']:.2%} | {row['net_sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['mean_exposure']:.2%} |")
    era=["| 时期 | 配对原点 | M1相对M0的MSE改善 |","|---|---:|---:|"]
    for row in pd.read_csv(OUT/"prediction_comparison.csv").itertuples():
        era.append(f"| {row.group} | {row.paired_origins} | {row.relative_mse_improvement:+.3%} |")
    econ=result["economic_increment"]["STRESS"]
    bounds={ref:[float(x)*242 for x in econ[ref]["paired_daily_increment_95pct_ci"]] for ref in econ}
    body=f"""# 解禁公告密度V1：完成检验，未建立可用优势

本轮完成免费原始来源取得、完整历史成分名单对齐、160次滚动拟合、1,604个评价预测（1,599个已成熟）和8个完整现金账户。压力费用下M1复合年化-2.649%、净夏普-0.246、最大回撤-31.77%；20万元终值167,456.35元。只交易510300与现金、年化10%及夏普1.2的共同目标仍未实现。

终态为 `{result['status']}`。本次用途已按事前规则否决，不再调20日窗口、反转方向、改变五日标签、挑年份或与旧策略拼接救援。资料取得成功和代码验证通过都不是策略通过。

## 真实来源和保留下来的失败

检索范围固定2015-01-01至2026-08-14，共140个自然月。巨潮原始“上市流通”检索共有33,722个唯一公告编号；17,745个符合固定标题规则，另有128个修订类标题单独保留。原始月分页得到33,722行但仅33,536个唯一编号，186个跨页重复遮住了186个缺失编号。源V1终态仍为NO_VIEW。

V1.1按日期分区修复77个失败月份，另外3个在启动时连接中断；原60个月完整来源复用，合计137个月完整。连接恢复后，单线程一次补齐其余3个月。跨批次还发现公告1225030021的接口时间相差1秒，编号、证券、标题、PDF及公告日一致；精确时间合并的V1.2程序因此失败，其失败回执保留。最后以另行记录的公告日粒度合同离线合并，140个月、33,722个唯一编号完整，旧编号无遗漏。

这里的“完整”是当前固定检索总体及分区声明一致，不是所有类型解禁的全市场全集，也不是历史HTTP首发证明。28,953条时间是00:00:00，另有4,769条包含时分秒；不能把它们混合当作严格日内首发。文件路径日期和上海公告日期全部一致。

四份原始PDF核对了三家公司。华友钴业2016年文件署期1月25日，而巨潮档案日期为1月26日，模型使用档案日期。立昂技术的一批股份名义解除限售76,613,628股，实际可上市流通17,938,638股；同日另一份公告有14,624,658股。金贵银业2024年文件为5,336,309股，不能用汇总表显示的2017年旧锁定承诺日期提前确认2024年事件。相关PDF、关键页图、汇总表原响应都在包内。

最初的东方财富ALL字段探测含个股解禁后涨跌字段，该响应作为未准入探测证据保存，没有进入本研究特征、标签或样本选择。后续模型只读公告日、证券、合格身份与独立历史成员名单，不读该响应。解禁不等于实际卖出，不能把它叫真实卖压。

## 唯一新增变量与时钟

公告日在上海时区加两个自然日后，才进入随后第一个交易日15:00原点的可用集合；策略最早在下一交易日09:30执行。延迟是明确的历史研究假设。以当前原点生效的300个成员为集合，统计过去20个交易原点内至少有一份合格公告可用的不同公司数，再除以300；窗口内一家公司只计一次。评价段变量取值0—7%，共22个不同数值。

成员来源的覆盖诊断中，合格公告涉及当时沪深300成员1,225份、1,211个公司日、421家公司。这个诊断用公告附近首个开市日识别成员；模型的实际时钟和成员取决于上段另行冻结的规则，二者不能混用。既有历史权重仍未准入，本轮没有使用。

M0是ETF 1/5/20日含分红收益、20日方差对数与自然月正余弦控制；M1只增加公告密度。2015-03-02起扩展训练，2020-01-02起评价，月首拟合，首模型有1,177个已成熟训练原点。相同训练样本、标准化、±5裁剪、岭罚0.1；训练五日标签退出日严格早于模型原点。2026-08-14行情末尾5个未成熟标签保留为CENSORED。

## 涨跌判断未形成有效排序

{chr(10).join(era)}

总体MSE改善为-0.3646%，三个时期全部变差；配对MSE差的95%区间为[-0.00000752641,0.00000102879]。方向准确率M0为47.97%，M1为48.22%；M1上涨召回率54.27%、下跌召回率42.72%。

M1判断未来五日成本后收益为正的893个原点，实际均值为-0.221%；判断非正的706个原点，实际均值为+0.181%。这些是预先报告的诊断，不构成事后翻转信号的许可。连续五日标签有重叠；63个交易日循环分块、2000次固定抽样不能消除项目多轮挑选的偏差。

## 完整账户结果

账户统一2020-01-02开盘至2026-08-11开盘清算，共1,601个记账行；8月12—14日三个不完整尾段交易日没有用于改善终值。初始20万元，T+1、100份整数手、价位、佣金最低5元、滑点、分红应收和到账均进入账户，现金及无风险收益均假设0。四个模型在基础和压力费用下都已实际计算，没有用预测显著性门提前挡住亏损账户。

{chr(10).join(table)}

压力账户M1相对M0的日收益差乘242为{econ['M0']['annualized_arithmetic_increment']:+.3%}，95%区间[{bounds['M0'][0]:+.3%},{bounds['M0'][1]:+.3%}]；相对仅波动控制为{econ['VOL_ONLY']['annualized_arithmetic_increment']:+.3%}，区间[{bounds['VOL_ONLY'][0]:+.3%},{bounds['VOL_ONLY'][1]:+.3%}]。它们是配对账户日收益差的年化算术量，不是两个复合年化相减，也不是中性化仓位回归alpha。只有一个时期同时优于两个对照，两个事前继续入口都失败。

## 验证和下一步

冻结时间为{result['freeze_created_at']}，唯一运行完成于{result['completed_at']}。新时钟/成员的4个合成边界测试通过；保存的160个模型、1,604行预测、8个账户和固定区块已重算，预测误差为0，未重新拟合、生成账户或抽样。来源重核还保留了pandas日期存储单位由秒变毫秒造成的严格dtype失败；补充只读核对将日期统一到纳秒后逐值严格比较，不修改输入。

下一步不能把更多“公告出现次数”当作有经济方向的信息，也不能把已有业绩预告修正、同机构同年度EPS修正或估值一致性研究重新包装为新候选。查重已发现上述旧研究及明确的前瞻EPS主线；应先读取它们的当前结果、失效原因和真实未覆盖数据，区分可补的来源缺口与已经否决的收益用途。这个后续分支本轮尚未运行，不能宣称已经找到了正确方向。

审阅应重点检查：公告日假设是否足够、当前成员与事件窗口有没有漏看或前视、季节控制是否合理、样本选择与重复试验偏差、方向诊断和完整账户是否一致，以及下一项工作究竟能增加什么新的证据。终态和本轮参数保持冻结。ZIP仅供本地核查或用户自行提交审阅，没有外部GPT反馈、部署或交易授权。
"""
    once(OUT/"研究结论与下一步.md",body)
    prompt="""请作为独立量化研究审阅者，完整审阅本包。目标是只交易510300和人民币现金，成本后复合年化至少10%、净夏普至少1.2，并要求独立验证。当前结果远未达标，不要为了用户目标美化亏损。

请先读00_README_FIRST.md、研究结论与下一步.md和冻结协议，再核对来源分区/失败回执、模型、逐日账户和保存重算。请具体回答：
1. 公告日期、延迟两日、成员资格、20原点窗口、五日标签成熟及分红现金时钟是否有实质错误？指出文件、字段和可检验的影响。
2. 186条原分页遗漏如何补回？一秒时间变化是否妥当限制为公告日用途？日期dtype的补充核对是否只解决存储精度而没有遮盖值差？
3. 为什么M1正预测组实际均值为负、负预测组实际均值为正？这是模型不足、信息无效还是可用时间假设问题？不要据此未经预先登记直接翻转信号或调窗口。
4. 八个账户是否真实遵守费用、T+1、整手、分红应收与到账、统一起止和终点清算？收益差区间是否被误读成独立alpha？
5. 来源结论、局部研究结论、项目达标和独立未来验证是否被清楚区分？指出多次研究选择偏差和本包仍不能证明的内容。
6. 在保留此冻结失败、旧IF/AH/EPU与既有EPS相关研究结论的前提下，给出下一步策略研究方向、优先级、要补的免费数据、最低验证标准和明确停止条件。已有业绩预告/EPS修正并非空白，建议须先说明相对旧用途的新证据是什么。

请给出质量批评和可执行下一步，不只评价打包是否完整。包的CRC、哈希或保存重算通过不等于外部科学验证、真实成交或高夏普策略成立。本包不授权订单、券商连接或实时交易。
"""
    once(OUT/"GPT审阅提示.md",prompt)


def main():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    saved=json.loads((OUT/"saved_verification_receipt.json").read_text(encoding="utf-8"))
    source_verified=json.loads((VERIFY/"source_saved_verification_v1_1_receipt.json").read_text(encoding="utf-8"))
    require(saved["status"].startswith("PASS_") and source_verified["exit_code"]==0,"保存重算未通过")
    require(not ZIP.exists(),"交付ZIP已存在，不能覆盖")
    plot_saved()
    write_reports(result)
    versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","pyarrow","requests","pytest","matplotlib","pdfplumber","pypdfium2"]}
    once(OUT/"requirements-review.txt","\n".join(f"{name}=={version}" for name,version in versions.items())+"\n")
    selected={}
    def add(path):
        path=Path(path)
        require(path.is_file(),"缺少打包文件："+str(path))
        selected[path.relative_to(ROOT).as_posix()]=path
    def tree(path):
        path=Path(path)
        require(path.is_dir(),"缺少打包目录："+str(path))
        for file in sorted(path.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                add(file)
    report_roots=[OUT,SOURCE,ROOT/"reports/research/510300_unlock_announcement_source_v1",ROOT/"reports/research/510300_unlock_announcement_source_v1_1",ROOT/"reports/research/510300_unlock_announcement_source_v1_2",VERIFY]
    raw_roots=[ROOT/"data/raw/market"/name for name in ["510300_unlock_source_feasibility_v1","510300_unlock_announcements_v1","510300_unlock_pagination_repair_probe_v1","510300_unlock_announcements_v1_1","510300_unlock_announcements_v1_2","510300_unlock_primary_pdf_cases_v1"]]
    extra_roots=[ROOT/"data/curated/510300_unlock_announcement_daily_source_v1",ROOT/"data/curated/510300_csi300_pit_membership_weights_source_remediation_v1",ROOT/"data/raw/510300_csi300_pit_membership_weights_source_remediation_v1"]
    for path in report_roots+raw_roots+extra_roots:
        tree(path)
    config=json.loads((ROOT/"config/510300_unlock_announcement_increment_v1.json").read_text(encoding="utf-8"))
    for rel in config["inputs"].values():add(ROOT/rel)
    freeze=json.loads((OUT/"freeze_manifest.json").read_text(encoding="utf-8"))
    for row in freeze["protected_files"]+freeze["input_snapshots"]:
        p=ROOT/row["path"]
        require(digest(p)==row["sha256"],"冻结对象变化："+row["path"])
        add(p)
    for pattern in ["*510300_unlock*.py","*unlock_announcement*.py"]:
        for folder in ["scripts","tests"]:
            for p in (ROOT/folder).glob(pattern):add(p)
    for pattern in ["510300_unlock_announcement_source*.json"]:
        for p in (ROOT/"config").glob(pattern):add(p)
    for p in (ROOT/"docs").glob("510300_UNLOCK_ANNOUNCEMENT*.md"):add(p)
    for rel in ["scripts/build_unlock_announcement_review_20260914.py","research/__init__.py","scripts/__init__.py","data/raw/r6/510300_daily.parquet","data/reference/510300_downside_risk_price_corrections_v1.csv","docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md","docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md"]:
        if (ROOT/rel).is_file():add(ROOT/rel)
    seeds=[ROOT/"data/reference/510300_dividends_coverage.json",ROOT/"reports/data_quality/510300_downside_risk_inputs_v1.json"]
    seeds += list((ROOT/"data/curated/510300_csi300_pit_membership_weights_source_remediation_v1").glob("*.json"))
    pending=list(seeds)
    visited=set()
    missing=[]
    def local_strings(value):
        if isinstance(value,dict):
            for v in value.values():yield from local_strings(v)
        elif isinstance(value,list):
            for v in value:yield from local_strings(v)
        elif isinstance(value,str):
            text=value.replace("\\","/")
            if text.startswith(("data/","reports/","config/","docs/","scripts/","research/")) and "\n" not in text:
                yield text
    while pending:
        path=pending.pop()
        if path in visited:continue
        visited.add(path)
        require(len(visited)<2000,"来源闭包超过预定预算")
        add(path)
        try:obj=json.loads(path.read_text(encoding="utf-8-sig"))
        except (UnicodeError,json.JSONDecodeError):continue
        for rel in local_strings(obj):
            candidate=ROOT/rel
            if candidate.is_file():
                add(candidate)
                if candidate.suffix.lower()==".json":pending.append(candidate)
            elif not candidate.is_dir():
                missing.append({"referring_file":path.relative_to(ROOT).as_posix(),"reference":rel})
    primary=json.loads((ROOT/"data/raw/market/510300_unlock_primary_pdf_cases_v1/20260914T002030_0800/download_receipts.json").read_text(encoding="utf-8"))
    for row in primary:require(digest(ROOT/row["raw_path"])==row["sha256"],"原始PDF哈希变化")
    previous=[]
    for name,expected in [("510300_IF持仓信息增量_V1_GPT审阅_20260913.zip","2a6672f5b7683488cc889d1e2cb9afbbfaa60afcae8838ed9531554887c71304"),("510300_AH溢价信息增量_V1_GPT审阅_20260913.zip","877ad6d86861d9005da955c840faa445418b5d4017eb1470ae22ae18205465fc"),("510300_EPU历史版本信息增量_V1_GPT审阅_20260913.zip","4c529c1dc8b8aaa99fece330a695ec629409976de9144f33ab5b27f7420cbbc6")]:
        p=ROOT/"deliverables"/name
        require(digest(p)==expected,"旧上下文ZIP身份变化")
        previous.append({"file":name,"bytes":p.stat().st_size,"sha256":expected,"included_in_new_zip":False,"expanded":False})
    metadata={"scope":"COMPLETE_CURRENT_UNLOCK_SOURCE_AND_FIXED_EXPERIMENT_WITH_FAILED_ATTEMPTS","created_at":now(),"study_status":result["status"],"source_roots":[p.relative_to(ROOT).as_posix() for p in raw_roots+extra_roots],"prior_archive_identities":previous,"recursive_source_metadata_files":len(visited),"unresolved_ancestor_references":missing,"direct_model_inputs_included":True,"all_current_announcement_raw_attempts_included":True,"unrelated_previous_research_full_archives_included":False,"runtime_environment_included":False,"byte_cap":512*1024*1024,"external_upload":False,"external_gpt_review":False,"security_audit":False}
    write_json(OUT/"package_scope.json",metadata,exclusive=True)
    add(OUT/"package_scope.json")
    require(all(row["raw_path"] in selected for row in json.loads((SOURCE/"admitted_response_provenance.json").read_text(encoding="utf-8"))),"有准入原响应未打入包")
    readme="""# 本轮阅读顺序

当前结论：解禁公告密度V1已否决，目标仍未实现。先读01_研究结论与下一步.md，再读02_GPT审阅提示.md、docs/510300_UNLOCK_ANNOUNCEMENT_INCREMENT_V1_PROTOCOL.md和冻结/结果文件。

目录保留项目相对路径。reports/research/510300_unlock_announcement_increment_v1包括全部模型、预测、8个账户、固定区块、输入快照及运行回执。data/raw/market/510300_unlock*包括来源探测、三批采集、连接失败、时间变化、原始PDF和关键页；不能只看最后的PASS而删掉旧失败。data/curated/510300_unlock_announcement_daily_source_v1是日粒度合并表。FILE_INDEX.csv为本ZIP唯一成员索引，索引本身不自引用哈希。

Windows复核：在新目录解压后，以Python建立独立虚拟环境，并安装reports/research/510300_unlock_announcement_increment_v1/requirements-review.txt中的版本。随后在解压根目录运行`.venv\\Scripts\\python.exe scripts\\verify_510300_unlock_source_saved_v1_1.py`，再运行`.venv\\Scripts\\python.exe scripts\\run_510300_unlock_announcement_increment_v1.py verify`。这两条只重核保存来源、特征、模型、账户及区块，不抓新数据、不重新拟合、不再模拟账户。后者会写验证回执，因此解压后的该回执时间会变化，原ZIP不变。

旧的verify_510300_unlock_source_admission_v1.py verify有已保留的pandas日期dtype失败；应使用上述补充只读入口，它统一日期存储精度后逐值严格比较。不要再次运行已经消费claim的采集、freeze或run入口。源代码保留是供审阅，不是要求重复研究。

包包含当前研究直接输入及原始来源、已有成员名单与来源证据；不包含Python运行环境和无关旧研究的全量数据。package_scope.json记载祖先来源引用与旧ZIP身份。原始历史权重文件即使作为成员来源证据收录，也没有准入为模型权重。结构检查、保存重算和本地打包不等于外部审阅或科学有效性，更不授权实时交易。
"""
    generated={"00_README_FIRST.md":readme.encode("utf-8"),"01_研究结论与下一步.md":(OUT/"研究结论与下一步.md").read_bytes(),"02_GPT审阅提示.md":(OUT/"GPT审阅提示.md").read_bytes()}
    index=[]
    for name,path in sorted(selected.items()):index.append({"path":name,"bytes":path.stat().st_size,"sha256":digest(path)})
    for name,data in generated.items():index.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    index.sort(key=lambda r:r["path"])
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=["path","bytes","sha256"])
    writer.writeheader();writer.writerows(index)
    temp=ZIP.with_name(ZIP.stem+".building.zip")
    require(not temp.exists(),"临时ZIP已存在")
    ZIP.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temp,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(selected.items()):archive.write(path,name)
        for name,data in generated.items():archive.writestr(name,data)
        archive.writestr("FILE_INDEX.csv",stream.getvalue().encode("utf-8-sig"))
    require(temp.stat().st_size<512*1024*1024,"压缩包超过512MiB上限")
    with zipfile.ZipFile(temp) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)),"ZIP成员重名")
        require(archive.testzip() is None,"ZIP CRC失败")
        require(set(names)=={row["path"] for row in index}|{"FILE_INDEX.csv"},"索引成员覆盖不符")
        for row in index:
            data=archive.read(row["path"])
            require(len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"],"成员哈希或字节数不符")
    os.replace(temp,ZIP)
    delivery={"status":"PASS_STRUCTURAL_CURRENT_RESEARCH_ZIP","created_at":now(),"path":str(ZIP),"bytes":ZIP.stat().st_size,"sha256":digest(ZIP),"members":len(index)+1,"indexed_members":len(index),"crc":"PASS","index_and_member_sha256":"PASS","raw_announcement_source_attempts_included":True,"source_saved_recomputation":"PASS_WITH_DATE_STORAGE_UNIT_NORMALIZATION","model_account_saved_recomputation":"PASS_NO_NEW_FITS_ACCOUNTS_OR_RANDOM_SAMPLES","unresolved_ancestor_references":len(missing),"external_upload":False,"external_gpt_review":False,"security_audit":False,"goal_achieved":False}
    write_json(ZIP.with_suffix(".delivery.json"),delivery,exclusive=True)
    print(json.dumps(delivery,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
