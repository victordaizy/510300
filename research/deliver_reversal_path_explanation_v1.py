"""整理回撤、反转、政策路径说明及自包含审阅交付。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_reversal_path_explanation_v1"
PARENT = ROOT / "deliverables/510300_财政执行状态与均值回归检验_V1_GPT审阅_20260922.zip"
PARENT_SHA = "0e4c5d28e3c62a2f4614222d3ee2163d145285034f5c14191d0eeefa822198c0"
ARCHIVE = ROOT / "deliverables/510300_回撤反转与政策路径说明_V1_GPT审阅_20260922.zip"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


def prepare():
    if ARCHIVE.exists():
        raise FileExistsError("归档已完成，不能覆盖")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("上一轮归档身份不一致")
    (OUT / "history").mkdir(exist_ok=True)
    shutil.copy2(PARENT, OUT / "history" / PARENT.name)
    save(OUT / "evidence/前阶段归档身份.json", {"path": PARENT.name, "sha256": PARENT_SHA, "checked_at": now()})
    for name in ["verify_reversal_path_explanation_v1.py", "deliver_reversal_path_explanation_v1.py"]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    rendering = OUT / "code/render_after_visual_labels_reversal_path_explanation_v1.py"
    shutil.copy2(ROOT / "research/reversal_path_explanation_v1.py", rendering)
    save(OUT / "evidence/图形标注修正.json", {"created_at": now(), "scope": "RENDERING_LABELS_ONLY_AFTER_SAVED_STATISTICS",
         "reason": "路径高点包含第0个开盘基点；全部收盘低于入场时，高点0不能标成最高收盘。将文字明确为路径高点/开盘基点。",
         "data_or_statistics_changed": False, "frozen_calculation_code_preserved": True,
         "rendering_code_sha256": sha(rendering)})
    policy = ROOT / "reports/research/510300_policy_information_clock_v1"
    nodes = pd.read_csv(OUT / "inputs/policy_nodes.csv")
    for record in nodes.drop_duplicates("source_path").itertuples():
        source = policy / record.source_path
        if sha(source) != record.source_sha256:
            raise ValueError("政策原文身份不同")
        target = OUT / record.source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    prior_copy = {
        ROOT / "reports/research/510300_original_quarterly_flow_policy_v1/result.json": "evidence/此前季度申赎结果_只读继承.json",
        ROOT / "reports/research/510300_money_consensus_increment_v2/研究结论.md": "evidence/此前货币预期与资金缺口_只读继承.md",
        ROOT / "reports/research/510300_fundamental_and_fund_flow_rebuild_v1/盈利估值股东回报与公募申赎_来源进展及口径.md": "evidence/此前全公募月报口径_只读继承.md",
        ROOT / "reports/research/510300_reversal_monthly_diagnostic_v1/figures/510300_二十日反转与延续对比.png": "figures/前轮_510300二十日反转与延续.png",
        policy / "figures/510300_货币数据与六类政策_完整走势.png": "figures/前轮_510300_M1M2与宏观政策走势.png",
        Path("E:/CodexData/.codex/attachments/9985585b-6e77-41bf-b927-0e02209a9c57/pasted-text-1.txt"): "用户原始目标.txt",
    }
    provenance = []
    for source, target in prior_copy.items():
        shutil.copy2(source, OUT / target)
        provenance.append({"origin": str(source), "copy": target, "sha256": sha(source), "claim": "继承保存结果或来源说明，不代表本轮重新运行其模型或审核全部底稿"})
    save(OUT / "evidence/继承材料身份.json", provenance)
    references = [
        {"id": "L1", "title": "Momentum or contrarian trading strategy: Which one works better in the Chinese stock market", "authors": "Lin Yu; Hung-Gay Fung; Wai Kin Leung", "year": 2019,
         "url": "https://www.sciencedirect.com/science/article/pii/S1059056018301928", "doi": "10.1016/j.iref.2019.03.006",
         "verified_scope": "本轮检索返回出版方文章摘要和方法说明；直接打开曾403，未读取完整付费论文。",
         "finding_paraphrase": "研究2010年以后个股赢家减输家的周收益，在沪市、深市和创业板观察到显著周度反转。",
         "boundary": "这是个股横截面多空组合证据，不等于510300自身固定二十日择时或费用后优势。"},
        {"id": "L2", "title": "Time series momentum and contrarian effects in the Chinese stock market", "authors": "Huai-Long Shi; Wei-Xing Zhou", "year": 2017,
         "url": "https://arxiv.org/abs/1702.07374", "publisher_url": "https://www.sciencedirect.com/science/article/pii/S0378437117304466", "doi": "10.1016/j.physa.2017.04.139",
         "verified_scope": "本轮核对作者摘要、出版方文章预览。",
         "finding_paraphrase": "中国主要指数的时间序列结果随回看和持有期限改变，摘要报告较短期动量与较长期反向效应。",
         "boundary": "原研究期限划分不能直接等同本轮二十个交易日；不能推出整个A股所有时点都应反转交易。"},
        {"id": "L3", "title": "Dissecting Momentum in China", "authors": "Xin Liu; Songtao Tan; Yuchen Xu; Peixuan Yuan; Yun Zhu", "year": 2025,
         "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5130681", "doi": "10.2139/ssrn.5130681", "version_date_seen": "2025-10-26",
         "verified_scope": "本轮检索返回作者工作论文摘要；直接打开403；未复算全文。",
         "finding_paraphrase": "作者将新闻日与后续非新闻日区分，提出暂时注意力买压和信息反应不足可能同时存在。",
         "boundary": "工作论文机制动机；没有提供本轮510300的实时压力分类或买卖信号。"},
        {"id": "SSE", "title": "上海证券交易所ETF常见问题", "url": "https://etf.sse.com.cn/fund/quertion/",
         "verified_scope": "本轮打开交易所原页面。", "finding_paraphrase": "股票ETF执行T+1；买卖最低100份；实物申购可用组合证券等对价创造份额。",
         "boundary": "份额增加不能单独识别净方向性资金意图；交易规则不建立收益优势。"},
    ]
    save(OUT / "sources/文献与官方资料核对.json", {"checked_at": now(), "references": references,
         "no_new_literature_backtest": True, "no_full_third_party_paper_redistribution": True})
    write("用户需求与本轮范围.md", """用户最新补充：‘还有一个问题，a股是一个均值回归，反转多于动量的市场’。

持续保留此前要求：用真实图形比较510300与M1/M2，纳入真实事前预期、其他宏观政策、市场状态、持续信息更新和真实资金行为；涨幅较大后结合回撤退出；20万元主账户、2万元成本对照。

本轮按已完成月末检验的固定170个原点展开每条20日路径，说明途中回撤与期末反转不是同一个量。补充文献适用对象，核对已有资金渠道的重复研究。它是事后描述，不再修改旧模型，也没有运行新的入场或退出账户。

未完成的整项研究仍然是：哪些当时可识别的信息或资金状态，能改变从当前价格继续持有的收益风险。路径图不能识别临时压力和基本面改变。宏观研究保持活动，旧策略终止和旧子实验失败继续保留。
""")
    write("00_README_FIRST.md", """# 阅读顺序

本包结论：A股反转值得作为条件性假设研究，但固定二十日510300数据没有证明反转比延续更多；上涨中回撤常见，也经常与期末上涨并存。回撤退出是否提高净收益仍未检验。

1. 研究结论.md：文献区别、170个原点、93个此前上涨路径、政策时钟及研究取舍。
2. figures/510300_途中回撤与期末反转.png：真实路径、完整交叉计数；下方两例按每类最早日期选取。
3. figures/510300_政策宣布与实施后的不同路径.png：同一政策链、不同可成交起点的说明；非因果试验。
4. results：170个月末路径、20条政策去重路径、3990个完整路径点和24个来源节点映射，CSV/Parquet双份。
5. protocol.json、freeze.json：事后描述口径；不借此次计算声称独立未见样本。
6. inputs、raw、sources、evidence：行情、既有固定结果、政策原文、文献范围、资金缺口与失败记录。
7. code/verify_reversal_path_explanation_v1.py：以--study-dir传入本包解压目录，复核保存数据，不运行策略。
8. GPT审阅提问.md：复制后连同完整ZIP交给审阅者。

history保存已先核验SHA-256的前轮完整ZIP，包含财政固定检验、原反转诊断及更早政策来源。FILE_INDEX.csv为本包成员身份表，不代表外部审阅或交易批准。
""")
    count = summary["counts"][1]
    write("研究结论.md", f"""# 510300：反转、回撤与宏观信息应怎样结合

**判断：A股存在值得研究的反转现象，但不能据此把510300设为‘涨了就卖、跌了就买’。已固定的二十日检验没有显示反转占优；进一步展开路径发现，途中回撤和最终上涨可以同时存在。用户提出的‘涨幅较大后，再根据回撤退出’，应作为持有期间的退出假设，不能单凭市场标签当作有效策略。**

行情继承2012-05-28至2026-09-11共3476个交易日；本轮使用2012-06-29至2026-07-31共170个月末原点。没有下载最新行情、训练模型、改变固定期限或运行账户。本轮新增路径统计是事后描述，独立前向事件为0。

## 文献支持的范围

Yu、Fung、Leung的研究在个股赢家减输家的周度组合中观察到反转；它比较股票之间的相对表现，不能直接迁移成一个宽基ETF的绝对择时。[出版方摘要](https://www.sciencedirect.com/science/article/pii/S1059056018301928)

Shi、Zhou的中国主要指数时间序列研究显示结果依赖回看与持有期限，并报告较短期动量和较长期反向效应。原研究的期限划分不能直接当成本轮二十日结论。[作者版本](https://arxiv.org/abs/1702.07374)

2025年的工作论文进一步区分新闻日与非新闻日，提出暂时注意力买压与消息反应不足可能同时存在。这支持把两种机制分开验证；它没有证明我们能实时识别510300哪一段应延续、哪一段应回归。[作者工作论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5130681)

本轮读取的是上述可访问摘要和文章预览，没有宣称复算其完整研究。当前网页可见版本不等于历史当时已知版本，文献仅作机制动机。

## 510300固定二十日结果

沿用前轮协议：每个完整自然月最后交易日，观察此前20日含分红回报；从下一开盘起计算随后20交易日现金权益回报。

|此前方向|随后上涨|随后下跌|随后持平|原点数|
|---|---:|---:|---:|---:|
|此前上涨|51|41|1|93|
|此前下跌|40|37|0|77|

全部原点中，81次反转、88次延续、1次零收益；非零原点中的反转比例47.93%，继承六个月连续区块的95%区间41.18%—55.03%。过去与未来回报相关系数约−0.0069，区间跨零。这里没有新增重采样或改变样本。**合理说法是固定设定下未证明反转占优；也没有证明动量占优。**

频数不等于交易胜率，更不等于费用后盈利；收益幅度、波动、尾部损失、空仓和执行都尚未计入。这个检验也没有估计合理价值或检验价格水平的平稳性，不能由此肯定或否定所有‘均值回归’命题。

![前轮固定诊断](figures/前轮_510300二十日反转与延续.png)

## 新展开的路径：回撤不等于期末转跌

此前上涨的93个原点中，{count['positive_peak_then_pullback']}次在后续20日先出现高于入场价的收盘现金权益高点，随后又回落。

|路径表现|次数|
|---|---:|
|途中回撤，期末仍上涨|{count['pullback_final_positive']}|
|途中回撤，期末转为下跌|{count['pullback_final_negative']}|
|途中回撤，期末持平|{count['pullback_final_zero']}|
|没有出现上述‘浮盈高点后回撤’|{count['no_positive_peak_pullback']}|

**任何很小的回落也会计入这一定义。**它说明路径上的回撤很常见，不能说明回撤足够大、能够交易，或卖出后比持有更好。全部170个原点的3570个路径点都保留，而非只挑两段形态画图。

![途中回撤与期末方向](figures/510300_途中回撤与期末反转.png)

下方两例分别是‘此前上涨、途中回撤、期末仍涨’和‘此前上涨、途中回撤、期末转跌’中日期最早的路径。高低点是事后注释，不能作为当时买卖价。路径按每份价格加现金分红权益计算，入场除息日没有分红权益；它不是净值账户，未扣交易费用。

## 政策与已经发生的涨幅，必须同时考虑

沿用前轮24个手工核实的政策来源节点，同一复核日合并为20条价格路径。源节点数不等于独立事件数；没有计算政策胜率。

2024年9月24日发布会宣布政策利率由1.7%拟降至1.5%，同时有降准等信息；9月27日实施是后续动作，不是重复的一次同等降息意外。[发布会原文](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)、[实施来源](https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html)

在本轮固定且保守的日频规则下，9月24日收盘观察后从9月25日开盘3.489元开始，随后20日最终现金权益收益为+15.02%。实施资料只有9月27日日期上界，首次可观察收盘顺延至9月30日，再逢国庆休市，执行起点为10月8日开盘4.656元；从该价格起的20日最终收益为−13.40%。第二条路径的最高点0是入场开盘基点，所有后续收盘低于它。

![政策不同进入时点](figures/510300_政策宣布与实施后的不同路径.png)

这不是最早可以交易的时钟，更不是比较两个最优买点；保守资料时间产生的延迟本身是结果的一部分。两窗口有重叠并包含其他消息，不能将全部收益归因于降息，也不能据此证明知道政策后就能辨认反转。它只展示：**政策内容、此前价格反应和当前成交价，共同决定需要评价的剩余持有区间。**

## 资金与下一项可证伪问题

本轮核对到，仓库已有正式季度申赎实验：7个候选、84次模型训练、16个账户，主候选基础成本年化约−1.51%、净夏普约−0.014；这些为保存结果继承，本轮未重跑。不能换个反转名称继续同一研究。

日份额1216行及2220条更早历史响应的公开时点缺口仍在；基金业协会77份月报还包含迁移、分类变化与前期修订问题。当前证据不足以将它们直接称为可实时观察的暂时抛压。股票ETF实物申购还可以交付股票篮子，份额增加本身不能识别净方向性资金意图。[交易所规则说明](https://etf.sse.com.cn/fund/quertion/)

可继续保留的两个独立机制是：新增长、贴现或政策信息改变持有理由后的持续调整；基本面消息不足以解释的暂时交易压力消散。前者要有消息及可观察市场反应，后者要有当时可辨认的真实压力证据。图形本身不提供这项分类，不能事后随涨跌改标签。缺资金资料只约束其对应模块，其他宏观通道继续独立推进。

用户的涨后回撤规则应先依附于固定入场，再比较卖出后持有现金与继续持有的共同起点收益和风险。执行从触发后可成交价格计算，保留T+1、跳空、分红和费用；不能以事后最高点代替卖价。当前入场信息增量尚未通过，因此退出比较、20万元主账户/2万元对照继续保持NOT_RUN，而不是填零或通过卖点微调挽救旧失败。

本轮完成路径与概念澄清；宏观、预期、资金和持续更新的整体研究仍未完成，也未建立年化10%、夏普1.2目标或独立前向证据。
""")
    write("GPT审阅提问.md", """请阅读00_README_FIRST.md及研究结论.md，重点批评‘A股均值回归、反转多于动量’是否被准确转成了可检验命题。

请核查：
1. 个股横截面反转、ETF时间序列反转、途中回撤、价格均值平稳和费用后交易利润是否被区分。
2. 170个原点的81反转/88延续/1零，以及93个此前上涨原点中84次浮盈后回撤、51次回撤后仍涨，是否正确计算和解释。任意微小回落的口径能否支持文中有限结论，是否被错误升级为交易机会。
3. 3990个保存路径点、入场日分红排除、次开盘时钟、20日结束、高低点事后标注是否一致；可运行只读复核器。
4. 两个政策窗口的保守时钟、节假日、信息重复与窗口重叠是否清楚；手工24来源节点是否被错误当成完整政策训练集。
5. 文献只有摘要/预览的阅读边界、既有季度资金研究失败、日份额时点缺口是否准确保留。不要把季度申赎失败外推为全部资金机制无效。

请给出下一项独立研究的经济机制、最少必要数据、当时可知时钟、固定对照、验收和停止条件。重点说明怎样在当时区分持续信息调整与临时资金压力；不要建议继续修改已失败家族的窗口、方向、样本或卖点，避免又收窄成仅一个预期差的固定五日模型。若某个新假设不依赖资金字段，请明确允许独立推进。

本包没有新增模型、策略账户、退出比较或独立前向事件。结构与数值复核不等于外部审阅，也没有交易、Paper、Shadow或实盘授权。
""")
    write("交付范围与排除说明.md", """本包包含当前全部路径、统计、CSV/Parquet、全部政策节点及对应原文、行情输入、冻结口径、源代码、只读复核代码、文献核对范围、用户要求、失败状态与图形。前轮ZIP先验SHA-256核验后原样保留，包含财政及原反转诊断所需资料。

文献仅留核对到的标题、来源、摘要事实转述及适用范围，不分发整篇付费或第三方论文。部分直接打开403，检索返回的出版方或作者摘要仍可用于上述限定说明，不被称为全文认证。

此前季度申赎结果及资金缺口报告为继承摘要，本轮未重跑其模型或复制大体量资金原始缓存，因此不将该部分称为新的数值验证。无最新行情下载、模型、回撤退出模拟、账户和订单。20万元和2万元只保留研究设定，没有填入不存在的账户指标。

冻结原始计算代码与目视核对后的文字修正版分别保存；修正版只改图形标签，没有改变路径或统计。code/verify_reversal_path_explanation_v1.py可在解压目录以--study-dir指定目录，依赖Python、numpy、pandas、pyarrow，只读取保存结果。研究脚本仍保留项目路径与一次运行保护，不承诺从任意目录重新运行研究。

排除无关工作区修改、虚拟环境、缓存、凭证、重复二进制供应商包，以及ZIP自身和外部交付回执。原始行情供应商全量下载包未重复纳入，本轮完整研究输入保留。FILE_INDEX.csv只建立成员身份；不能替代科学验证或外部审阅。
""")
    print("研究结论、文献边界、全部数据和审阅导航已完成。", flush=True)


def package():
    if ARCHIVE.exists():
        raise FileExistsError("最终归档已存在")
    if not (OUT / "evidence/图表目视核对.json").exists():
        raise ValueError("尚未记录图表目视核对")
    command = [sys.executable, str(OUT / "code/verify_reversal_path_explanation_v1.py"), "--study-dir", str(OUT)]
    checked = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8").stdout)
    save(OUT / "evidence/保存路径独立复核.json", checked)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"FILE_INDEX.csv", "delivery_receipt.json"} and "__pycache__" not in p.parts and p.suffix != ".pyc")
    entries = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(entries)
    index = buffer.getvalue().encode("utf-8-sig")
    building = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for p in files:
            archive.write(p, p.relative_to(OUT).as_posix())
        archive.writestr("FILE_INDEX.csv", index)
    if building.stat().st_size > 80 * 1024 * 1024:
        raise ValueError("超过本轮80MiB范围")
    with tempfile.TemporaryDirectory(prefix="reversal_path_verify_") as temp:
        with zipfile.ZipFile(building) as archive:
            names = archive.namelist()
            if archive.testzip() is not None or len(names) != len(set(names)):
                raise ValueError("ZIP结构检查失败")
            if set(names) != {r["path"] for r in entries} | {"FILE_INDEX.csv"}:
                raise ValueError("索引与成员不一致")
            for row in entries:
                data = archive.read(row["path"])
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("索引字节数或哈希不同")
            archive.extractall(temp)
        result = subprocess.run([sys.executable, str(Path(temp) / "code/verify_reversal_path_explanation_v1.py"), "--study-dir", temp], check=True, capture_output=True, text=True, encoding="utf-8")
        if json.loads(result.stdout) != checked:
            raise ValueError("新解压目录复核结果不同")
    building.replace(ARCHIVE)
    (OUT / "FILE_INDEX.csv").write_bytes(index)
    receipt = {"created_at": now(), "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
               "sha256": sha(ARCHIVE), "members": len(entries) + 1, "indexed_members": len(entries),
               "status": "PASS_STRUCTURAL_AND_SAVED_PATH_RECOMPUTATION", "crc": "PASS", "duplicates": 0,
               "index_size_hash": "PASS", "fresh_extraction_verification": checked,
               "external_review_completed": False, "new_models": 0, "new_accounts": 0,
               "whole_macro_objective_complete": False}
    save(ARCHIVE.with_suffix(".receipt.json"), receipt)
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="路径说明完整研究交付")
    parser.add_argument("action", choices=["prepare", "package"])
    {"prepare": prepare, "package": package}[parser.parse_args().action]()
