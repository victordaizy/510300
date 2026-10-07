"""整理预期差扩样研究、微信资料核查和离线审阅包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
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
STUDY = ROOT / "reports/research/510300_money_consensus_increment_v2"
EXT = ROOT / "reports/research/510300_money_consensus_source_extension_v1"
PARENT = ROOT / "reports/research/510300_post_information_capital_adjustment_v1"
ARCHIVE = ROOT / "deliverables/510300_货币预期差扩样与五日增量_V2_GPT审阅_20260921.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare() -> None:
    parent_zip = ROOT / "deliverables/510300_消息后剩余行情与资金增量_V1_GPT审阅_20260921.zip"
    expected = "51e33baf45dbd8b8c9eb8e3dfde7a3b77f05b92b5da024556ef1b2701c8bb150"
    assert digest(parent_zip) == expected
    selected = ["用户原始附件.md", "研究结论.md", "results/summary.json", "results/20万与2万_整手费用示例.csv", "results/六次事件_观察后五日完整结果.csv", "evidence/policy_context_source_receipts.json", "evidence/policy_data_receipt.json", "evidence/prior_policy_events.csv", "evidence/NBS_G2_adjudication.json", "evidence/research_authority_snapshot.json", "evidence/prior_input_origins.json", "sources/104个月原始页面定位.csv"]
    identity = []
    with zipfile.ZipFile(parent_zip) as z:
        for rel in selected:
            content = z.read(rel)
            assert content == (PARENT / rel).read_bytes(), rel
            if rel == "sources/104个月原始页面定位.csv":
                target = STUDY / rel
            else:
                target = STUDY / "evidence/parent_snapshot" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            identity.append({"parent_member": rel, "sha256": hashlib.sha256(content).hexdigest()})
        for rel in z.namelist():
            if rel.startswith("sources/official_money/") and not rel.endswith("/"):
                target = STUDY / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(rel))
    write_json(STUDY / "evidence/parent_archive_identity.json", {"filename": parent_zip.name, "sha256": expected, "checked_selected_members": identity, "role": "前轮快照保留，其六例状态不代表本轮扩样后的覆盖数量。"})
    official = pd.read_csv(STUDY / "sources/104个月原始页面定位.csv")
    for row in official.itertuples():
        assert digest(STUDY / row.package_source_path) == row.source_sha256

    for family, folder in [("Bualuang", EXT), ("OCBC", EXT / "ocbc"), ("NBG_FIVE_DAY", EXT / "nbg"), ("NBG_WEEKLY", EXT / "nbg_weekly")]:
        target = STUDY / "sources/search_history" / family
        for f in folder.glob("*.json"):
            if f.name != "progress.json":
                copy(f, target / f.name)
        for sub in ("inputs", "results", "receipts", "code"):
            if (folder / sub).exists():
                for f in (folder / sub).iterdir():
                    if f.is_file() and f.suffix in (".csv", ".json", ".py"):
                        copy(f, target / sub / f.name)
    for f in (EXT / "fund_clock").glob("*.json"):
        copy(f, STUDY / "sources/fund_clock" / f.name)
    fund_inventory = json.loads((EXT / "wechat/inventory.json").read_text(encoding="utf-8"))
    # 交付仅保留来源定位和研究所需有限数字，不复制整篇报告或长篇摘要。
    clean_inventory = [{k: v for k, v in row.items() if k != "cover_excerpt_for_internal_qa"} for row in fund_inventory]
    write_json(STUDY / "sources/wechat/检索及原文获取回执.json", clean_inventory)
    for f in (EXT / "wechat/extracts").glob("sogou_query_*.json"):
        copy(f, STUDY / "sources/wechat" / f.name)
    fund_records = [
        {"key": "SOGOU_WECHAT_SEARCH", "status": "SEARCH_SNIPPETS_ONLY", "finding": "首次公开微信检索有10条文章线索；随后3个查询返回验证码页面，不代表没有文章。未取得对应mp.weixin正文，不将摘要准入为数据。"},
        {"key": "ETF_WANYI_SYNDICATION", "url": fund_inventory[1]["url"], "page_published_at": "2025-06-12T12:00:00+08:00", "attribution": "页面标注转自ETF万亿指数", "status": "SYNDICATED_PAGE_READ_ORIGINAL_WECHAT_NOT_READ", "limitation": "主要内容为图片；标题超10亿是金额描述，未得到完整五日份额变量。"},
        {"key": "HUATAI_ETP_20201220", "url": fund_inventory[2]["url"], "sha256": fund_inventory[2]["sha256"], "cover_date": "2020-12-20", "reported_week": "2020-12-12至2020-12-19", "page5_510300_weekly_share_change_pct": -3.18, "page5_fund_size_100m_cny": 381.55, "page11_weekly_share_change_pct": -3.18, "page11_fund_size_100m_cny": 441.52, "page11_labeled_fund_flow_100m_cny": -4.53, "status": "NUMERIC_FACTS_READ_VISUALLY_CHECKED_NOT_ADMITTED_TO_FIXED_FUND_FEATURE", "limitation": "两处份额变化率一致，规模列相互冲突；单个自然周与各宏观观察日的最近五交易日不等价，首次公开时刻未独立证明。"},
        {"key": "HUABAO_ETP_20190710", "url": fund_inventory[3]["url"], "sha256": fund_inventory[3]["sha256"], "cover_writing_date": "2019-07-10", "url_date": "2019-07-11", "page3_510300_size_change_100m_cny": -1.89, "page3_510300_latest_size_100m_cny": 356.82, "status": "SIZE_CHANGE_NOT_NET_SUBSCRIPTION", "limitation": "该数位于规模变化列；规模变化包含价格变化，不能当作净申赎。前十排名表也不是全基金连续日序列。"},
        {"key": "HUABAO_ETP_201903", "url": fund_inventory[4]["url"], "sha256": fund_inventory[4]["sha256"], "cover_writing_date": "2019-03-04", "url_date": "2019-03-06", "page3_510300_previous_day_share_change_10k_units": -15750.0, "page3_510300_size_change_100m_cny": -1.90, "status": "TRUE_SHARE_CHANGE_FACT_FOUND_DATE_AND_COVERAGE_UNRESOLVED", "limitation": "原表确有T-1份额变化，不应说没有资金数据；但封面与URL日期不同，尚未确认对应交易日及首次发布时间，也缺连续五日和期初份额。"},
        {"key": "CHINA_SECURITIES_JOURNAL_20200319", "url": fund_inventory[5]["url"], "status": "TWO_DAY_MONEY_FLOW_CONTEXT_ONLY", "limitation": "原报表为本周前两个交易日资金净流入金额，不能替换冻结的最近五日份额变化率。"},
    ]
    write_json(STUDY / "sources/wechat/资金字段与准入裁决.json", fund_records)

    scripts = ["money_consensus_increment_v2.py", "verify_money_consensus_increment_v2.py", "plot_money_consensus_increment_v2.py", "deliver_money_consensus_increment_v2.py", "collect_bualuang_money_consensus_v1.py", "collect_ocbc_money_consensus_v1.py", "collect_nbg_money_consensus_v1.py", "complete_nbg_weekly_consensus_v1.py", "check_fund_clock_scope_extension_v1.py", "probe_wechat_fund_sources_v1.py"]
    for name in scripts:
        copy(ROOT / "research" / name, STUDY / "code" / name)
    requirements = [f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "pandas", "pyarrow", "matplotlib", "pdfplumber", "pypdfium2", "requests", "tzdata")]
    (STUDY / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    summary = json.loads((STUDY / "results/summary.json").read_text(encoding="utf-8"))
    a = summary["adjudications"][0]
    same = json.loads((STUDY / "results/描述统计.json").read_text(encoding="utf-8"))
    source_counts = pd.read_csv(STUDY / "inputs/104个月共识选择.csv")
    missing = source_counts[~source_counts.consensus_admitted]
    missing.to_csv(STUDY / "results/27个月缺失配对预期.csv", index=False, encoding="utf-8-sig")
    report = f"""# 510300：补齐事前预期后，是否还有消息增量

**结论：本轮已把配对预期从此前六个历史案例扩展到77个月，并完成旧M1口径的固定增量检验。加入剪刀差预期差后，五日收益预测MSE比价格基准高4.98%，前后两半都没有改善，冻结拒绝这项具体实现。新M1口径16个成熟案例中，首轮反应14次与预期差同向，之后五日仅8次同向；这只能描述样本，不能形成交易依据。政策和真实资金仍有数据缺口，不能声称所有信息已补齐或账户已达标。**

本轮本金口径：20万元为主，2万元仅作成本对照。行情冻结为2012-05-28至2026-09-11共3476个交易日；宏观所属月2018-01至2026-08共104个月。研究日期2026-09-21不表示行情已更新至当日。

## 1. 先看用户要求的对比图

![价格、实际与预期](figures/510300_M1M2_实际与预期走势对比.png)

四行共享时间轴：510300未复权收盘价、M1和M2同比、实际剪刀差与事前预期、实际减预期。宏观线在实际公布时刻才跳变，不提前放到所属月末。预期点放在对应实际公布日便于比较；其原始报告日期另存，不表示预期当天才出现。

剪刀差定义为M1同比减M2同比，单位百分点。预期差为(M1实际−M1调查预期)−(M2实际−M2调查预期)。两项调查汇总值相减，是剪刀差预期的代理，并不等于同一批受访者逐一计算后的剪刀差中位数。各自的M1/M2预期差也完整保存。

2025年新M1口径单列，首次公布位置在图中标红；没有把回溯修订的2024新口径数值冒充当时信息。2012年起的完整日线另见第四张图，宏观准入数据从2018年开始。全部日线CSV/Parquet保留，不下采样。

## 2. 来源扩展确实带来了新数据

| 来源 | 该来源合格配对 | 按冻结优先级新增 | 说明 |
|---|---:|---:|---|
| Bualuang原站日报 | 61 | 61 | 44旧口径、17新口径 |
| OCBC原站日报 | 14 | 14 | 补齐较早月份 |
| NBG原五日检索 | 5 | 1 | 其余与优先来源重复 |
| NBG自然周报补缺 | 1 | 1 | 2023-08所属月，公布前6天报告 |
| 合并 | — | 77 | 60旧口径、17新口径；剩余27个月缺失 |

最初按公布前1—5个自然日寻找日报和周报。在旧口径达到59个月、尚未读取本轮新增收益时，另行冻结全部28个缺失月份的自然周报检索：仅取实际公布前最近一个星期二，间隔1—7天，不取同日或更早一期。全部48个候选地址检索完成后只增加1个月。没有找到第60个月就提前停止，也没有根据收益选择来源。但这一来源扩展本身仍属于上游研究选择，必须保留，不能称为完全未经探索的试验。

最终来源优先级Bualuang→OCBC→NBG原检索→NBG自然周报，在收益计算前固定。准入需同时满足中国M1/M2对应月份、同一报告配对、封面日期早于实际公布、实际列尚空、两项前值与上一月官方值相符。重复来源也保存。实际选中报告距公布1—6天（检索上限7天），不保证是公布前最后一刻的调查预期。

示例原文：[Bualuang 2024-06-12](https://research2.bualuang.co.th/upload/Bls240612.pdf)、[OCBC 2019-02-14](https://www.ocbc.com/assets/pdf/daily%20treasury%20outlook/2019/dto%2014022019.pdf)、[NBG 2023-09-05](https://www.nbg.gr/-/jssmedia/Files/Group/meletes-oikonomikes-analuseis/diethneis-agores-oikonomia/ebdomadiaia-episkopisi-diethnwn-agorwn/NBG-GlobalMarketsRoundup-05-09-2023.pdf)。每个月的URL、报告日期、页码、关键数字、原文件哈希在sources/selected_numeric_sources.json。

这些是今天取得的带日期原报告，可用于历史重建；当前哈希不能证明历史版本从未修改。没有建立不可变的事前采集，也没有增加独立前向事件。

## 3. 固定A/B检验结果

观察时钟：公布后的第一个完整交易日收盘，记录公告前20日总回报、首轮反应与截至当时已知20日波动。标签从下一交易日开盘起，到第5交易日预定收盘止。此前跳空和首日涨幅不计入可参与收益。训练只使用退出时刻已早于当前观察时刻的标签。

A使用上述三个价格变量；B仅新增一个剪刀差预期差。旧、新M1口径分开。固定ridge，训练内标准化，惩罚为1，不搜索参数。最低36个训练事件、24个评价事件；这是执行门槛，不是统计功效保证。

| 指标 | A：价格基准 | B：加预期差 |
|---|---:|---:|
| 评价事件 | 24 | 24 |
| MSE（小数收益平方） | {a['MSE_A']:.10f} | {a['MSE_B']:.10f} |
| RMSE（五日收益百分点） | {a['MSE_A']**.5*100:.3f} | {a['MSE_B']**.5*100:.3f} |
| B相对A的MSE变化 | — | +4.983%（变差） |

评价成交期2022-12-14至2025-01-22，对应2022-11至2024-12的24个合格所属月；中间不合格月份仍缺失。每个评价点扩展训练，未把后来的标签提前使用。

A减B平方误差均值为{a['MSE_improvement_mean']:.8f}；按固定4事件连续移动块、10000次、种子5103005得到的90%单侧下界为{a['MSE_improvement_one_sided_90pct_lower']:.8f}，也小于零。前半、后半平均差分别{a['evaluation_half_improvements'][0]:.8f}和{a['evaluation_half_improvements'][1]:.8f}。三项预定增量条件均未通过，结果为REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT。

![全部评价及新口径案例](figures/预期差_增量检验与首轮后续对照.png)

旧口径60个历史事件：19次正预期差、38次负预期差、3次零预期差；首轮同向29次，后续五日同向30次。新口径有17个月配对预期，其中16个标签成熟、1个受行情截止限制未成熟；首轮同向14/16，后续五日8/16。零涨幅不记同向。这些计数不是预测胜率，不与旧口径混成一个总体显著性结论，也不支持反向交易。

新口径未达到60个事件，模型保持NOT_RUN_SAMPLE_GATE。不能把旧口径失败推广成所有时期、所有宏观预期变量永远无效；能拒绝的是本轮事前固定的数据代理、模型和五日实现。

## 4. 政策如何纳入

![政策背景](figures/510300_预期差与政策背景.png)

2024年9月24日降准降息与资本市场工具、9月26日政治局部署，和随后510300急涨在时间上邻近，而当时已公布的剪刀差仍处于低位。这说明仅看剪刀差水平不足以完整描述行情；图形不能证明上涨全由这两项政策造成。宣布日、实施日和首次披露时刻需要区分，图中八项事件均保存官方或原始来源。比如[9月24日发布会](https://www.csrc.gov.cn/ningbo/c101607/c7509061/content.shtml)包含多项措施，不能把它压成一个没有事前基准的“利好”哑变量。

八项事件是原研究的背景清单，不是完整政策样本；政策预期和意外程度字段仍为空。此前446条政策记录一并保留，但公开Target/Path因子使用全样本PCA，不能直接放进逐期交易预测。需要同类政策完整清单、当时可知的预期、原始金融价格和逐期估计，才可检验政策增量。参考[作者说明](https://harrisonshieh.com/research/)。本轮没有用本次已知涨跌反过来编码政策强弱。

## 5. 微信公众号、转载及原研报查到了什么

微信公众号方向已实际搜索。首个搜狗微信结果页取得10条文章线索；后续3个查询遇到验证码，没有绕过。搜索条目不等于已读原文。还沿公开转载取得ETF万亿指数栏目页面和三份原始证券研究报告，来源回执及字段裁决见sources/wechat。

| 查到的材料 | 可核实事实 | 尚不能替代的内容 |
|---|---|---|
| ETF万亿指数的新浪转载 | 2025-06-12 12:00页面、转自标记及标题 | 原公众号全文、完整五日份额序列 |
| 华宝证券2019年3月ETP日报 | 第3页510300的T-1份额变化为−15750万份 | 封面2019-03-04与URL2019-03-06不同；所属交易日、首次公开时刻和完整连续覆盖未确认 |
| 华宝证券2019年7月ETP日报 | 第3页510300规模变动−1.89亿元 | 这是规模变化，不是纯申赎金额；包括价格影响 |
| 华泰证券2020-12-20周报 | 第5、11页份额变化均为−3.18% | 两表规模分别381.55与441.52亿元，存在原表冲突；自然周也不等于每个事件所需五日 |
| 中国证券报2020-03-19原报 | 两个交易日的资金流金额表 | 最近五日份额变化率和期初份额 |

原文：[华宝3月报告](https://pdf.dfcfw.com/pdf/H3_AP201903061303140553_1.pdf)、[华宝7月报告](https://pdf.dfcfw.com/pdf/H3_AP201907111338430984_1.pdf)、[华泰周报](https://crm.htsc.com.cn/doc/2020/10750401/b1a76cca-5e88-45c7-a48a-6f3592b829a4.pdf)、[ETF万亿指数公开转载](https://finance.sina.com.cn/roll/2025-06-12/doc-inezutkh1583924.shtml)。关键表格已经逐页核对，不把未入榜的基金变化记为零。

上交所网站所说23:00更新对应全市场汇总规模；旧通知的8:00规则对应PCF申赎清单。两者都没有证明历史每个510300份额字段何时首次公开。旧来源核查有1216行数值和2220条历史响应，这次保留其缺时点结论，没有重复抓取后把抓取时间改称公布时间。

因此，本轮已证明公开材料中存在真实份额信息；仍未建立足量事件的冻结资金变量“观察时点已知的最近五交易日份额净变化/期初份额”。C、D保持NOT_RUN_FUND_CLOCK_AND_COVERAGE。成交量、规模变化、主力净流入标签不能替代该变量。

## 6. 回撤退出与本金

用户提出的“涨幅较大，再根据回撤退出”已保留为一个固定候选：入场前已知20日波动为σ，持有收盘相对入场上涨达到2σ后启动，从持有期收盘高点回撤达到σ时次日开盘退出，最晚仍第5日预定收盘。参数没有最优性证据。本轮入场尚未通过，退出比较保持NOT_RUN，不能靠改卖点补救失败的消息增量。

20万元为主、2万元作成本对照的费用示例完整保留：每边万二且最低5元、100份整手，基础/压力滑点每边5/10bp。以4元不变参考价，20万满预算基础/压力往返约需0.140%/0.240%涨幅覆盖成本；2万满预算约0.151%/0.251%，2万只用25%预算约0.309%/0.409%。这是费用算例，不是信号账户。

本轮没有生成任何账户净值；年化收益和净夏普为未计算，不填零。完整账户10%年化、1.2净夏普没有达成证据。现金天数、分红到账、T+1和费用等需要在合格入场后由同一账户引擎处理。波动本轮只作风险与控制变量，不宣称改进风险预测。

## 7. 核对、限制与下一步取舍

独立程序核对了77份数值配对、3476日日收益与总回报等式、76个事件的时钟和标签；使用增广最小二乘重算48组固定模型，最大预测差约1.03e−16，固定种子重建区块统计。模型结果与首次成功运行一致。图表已检查标签、口径切换、缺失预期、所有每日点及政策编号。

保留两个实现修正：时区最小时间溢出在首次拟合前修复；两项理论为零的十进制差因浮点变成−1.78e−15，仅描述性符号统计设1e−10容差。后者第一次文字替换遇到CRLF导致描述阶段报错，随后修正。协议、训练特征、原始输入和预测输出没有更改；相关原代码与回执均保存。

DSR没有完整上游试验分布及新账户，保持NOT_COMPUTED。旧85/15终止不变，NBS此前G2已经失败，不重启。附件关于NBS尚未运行G2的说法已由原裁决更正，并保留证据。

**当前取舍：保留M1/M2、预期和政策作为背景；停止本轮失败的预期差交易实现，不改窗口、方向或来源优先级。只有独立补足历史发布时间和连续覆盖的真实资金资料，或可逐期构造的政策意外，才有理由另行预注册检验。数据来源可继续积累，但当前不产生买卖结论。**

这份交付完成了来源扩展、图形对比和本轮可执行的A/B判定；不代表所有政策与资金信息都已补齐。新M1口径增加历史来源也无法凭空增加2025年至今的月份数，不能与旧口径拼接凑门槛。严格前向事件仍为0。
"""
    (STUDY / "研究结论.md").write_text(report, encoding="utf-8")
    (STUDY / "用户需求与本轮口径.md").write_text("用户要求510300与M1/M2走势对比，重点考虑是否超出事前市场预期及宏观政策影响；补齐可取得资料后给出结论。最新明确允许网上查询，尤其微信公众号。20万元为主、2万元作成本对照；涨幅较大后根据回撤退出。原始长附件保存在evidence/parent_snapshot/用户原始附件.md。本轮继承固定五日协议，先验证入场信息，失败后不调参数救援。\n", encoding="utf-8")
    (STUDY / "00_README_FIRST.md").write_text("# 阅读顺序\n\n先读研究结论.md及四张figures图；再读protocol.json和results/summary.json；随后检查inputs、sources与evidence。旧口径B的MSE比A高4.98%，本轮拒绝；新口径样本不足，C/D资金资料未准入。20万与2万账户、回撤退出均未运行。\n\n完整104个月选择表、27个月缺失表、76个成熟事件、24次评价与48组模型均已保留。来源是历史重建，不是严格前向证据。来源扩展经过四个固定阶段，全部记录在sources/search_history。\n\n在Windows PowerShell安装requirements.txt依赖后，从解压根目录可执行以下命令（脚本位于code目录）：\n\n- 保存结果核对：`python code/verify_money_consensus_increment_v2.py --study-dir .`\n- 重算相同历史模型至新目录：`python code/money_consensus_increment_v2.py analyse --study-dir . --output-dir replay_results`\n- 重绘图表：`python code/plot_money_consensus_increment_v2.py --study-dir . --output-dir replay_figures`\n\n第一项不下载、不新增研究候选；重算模型只使用本包冻结输入。原报告整篇PDF未附，无法离线重做PDF解析；已附必要数字、页码、URL和哈希，外部原文件可能变更。来源抓取代码用于记录原流程，路径依赖原仓库，不属于上述离线重算命令。FILE_INDEX.csv覆盖除自身外全部包成员。未进行外部GPT评审。\n", encoding="utf-8")
    (STUDY / "GPT审阅提示词.md").write_text("请独立审阅本包510300货币预期差研究。先读结论、protocol、freeze_receipt和summary，再核对来源及全部评价。请区分本地保存证据、原报告的历史重建、不可核验的事前版本和独立前向证据。重点检查：来源优先级与自然周报1—7天扩展是否增加选择偏差；M1/M2边际调查之差的经济含义；2025口径断点；公布时钟和标签成熟；48组固定ridge及4事件区块检验；旧口径失败和新口径不足的解释边界。不要因最终失败而反向选择新窗口或方向。\n\n请检查微信公众号摘要、公开转载与原始全文的区分，华宝T-1份额与规模变化、华泰原表冲突，以及C/D没有足量连续资金字段的状态。政策八项为背景，不是完整事件样本，446条旧政策因子存在全样本PCA限制。\n\n请给出事实错误、计算错误、识别问题、最有价值的下一项策略方向、优先级、所需新证据、预注册验证方法及停止条件；说明哪些建议仅为假设。不要把结构/复算通过称为科学有效或外部已评审。20万元主账户、2万元对照未运行，年化10%/夏普1.2未实现，退出也未运行。本包不授权订单或实盘。\n", encoding="utf-8")
    (STUDY / "排除清单.md").write_text("# 未包含内容\n\n- 未复制受版权保护的整篇证券研究PDF、完整解析文字或内部QA全页截图；保留研究所需数值、页码、URL、来源哈希。104份央行统计页面保留。\n- 未复制验证码原始HTML、网页跟踪参数和无关整站HTML；保留访问状态及原回执摘要。验证码不被绕过。\n- 未复制整个仓库或旧ZIP；旧包SHA-256已核对，直接相关快照逐成员核对后纳入。\n- 没有C/D模型、20万/2万账户或回撤退出结果，因为准入未通过；不是文件遗漏。\n- 没有补造27个月预期、资金公开时刻、政策意外或严格前向记录。\n- 完整上游试验分布、不可变历史预期版本、原政策逐期价格因子、全事件资金序列仍不可得，DSR未计算。\n", encoding="utf-8")
    coverage = [
        ["510300走势与M1/M2比较", "完成", "四张图、全日频CSV/Parquet；宏观2018起，行情2012起"],
        ["事前预期", "部分补齐", "104个月中77个配对，27个仍缺失；历史版本非不可变"],
        ["旧口径消息增量", "完成并拒绝", "60事件、24评价，B预测误差增加4.98%"],
        ["新口径增量", "NOT_RUN_SAMPLE_GATE", "17个月配对，16成熟；低于60事件门"],
        ["政策影响", "背景图完成，预测未建立", "8背景事件、446旧记录；缺完整事前政策意外与逐期因子"],
        ["微信与资金", "新事实已核对，C/D未准入", "份额数字存在，连续五日/期初份额/历史公开时点不齐"],
        ["20万主账户与2万对照", "NOT_RUN", "保留费用示例；入场和D门未通过"],
        ["上涨后回撤退出", "NOT_RUN_ENTRY_NOT_VALIDATED", "规则冻结但未比较收益"],
        ["独立前向证据", "0", "全部是历史开发；未证明年化10%或净夏普1.2"],
    ]
    pd.DataFrame(coverage, columns=["用户需求", "状态", "证据与边界"]).to_csv(STUDY / "需求完成与缺口.csv", index=False, encoding="utf-8-sig")
    write_json(STUDY / "status.json", {**summary, "research_round_status": "COMPLETED_AVAILABLE_SOURCE_RESEARCH_WITH_EXPLICIT_GAPS", "main_capital": 200000, "cost_comparison_capital": 20000, "position_impact": 0, "external_GPT_review": "NOT_PERFORMED"})
    print("研究结论、微信字段裁决、来源历史和离线重算材料已整理。")


def package() -> None:
    assert (STUDY / "visual_check.json").exists()
    index = STUDY / "FILE_INDEX.csv"
    members = sorted(p for p in STUDY.rglob("*") if p.is_file() and p.name != "FILE_INDEX.csv" and "__pycache__" not in p.parts)
    with index.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for p in members:
            writer.writerow({"path": p.relative_to(STUDY).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)})
    stage = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(stage, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in [*members, index]:
            z.write(p, p.relative_to(STUDY).as_posix())
    with zipfile.ZipFile(stage) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        rows = list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert set(z.namelist()) == {r["path"] for r in rows} | {"FILE_INDEX.csv"}
        for row in rows:
            blob = z.read(row["path"])
            assert len(blob) == int(row["bytes"])
            assert hashlib.sha256(blob).hexdigest() == row["sha256"]
        with tempfile.TemporaryDirectory(prefix="money_v2_delivery_") as tmp:
            target = Path(tmp)
            z.extractall(target)
            proc = subprocess.run([sys.executable, str(target / "code/verify_money_consensus_increment_v2.py"), "--study-dir", str(target)], check=True, text=True, encoding="utf-8", capture_output=True)
            recomputed = json.loads(proc.stdout)
            replay = target / "replay"
            subprocess.run([sys.executable, str(target / "code/money_consensus_increment_v2.py"), "analyse", "--study-dir", str(target), "--output-dir", str(replay)], check=True, text=True, encoding="utf-8", capture_output=True)
            result_files = ["全部准入事件_固定五日.csv", "已准入但行情未成熟.csv", "A_B全部逐期预测.csv", "仅当预测门通过的逐事件成本检验.csv", "全部模型训练记录.json", "分口径裁决.json", "描述统计.json", "summary.json"]
            for name in result_files:
                assert (replay / name).read_bytes() == (target / "results" / name).read_bytes(), name
    stage.replace(ARCHIVE)
    receipt = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "PASS_STRUCTURAL_CHECKS_FRESH_EXTRACTION_AND_FROZEN_RECOMPUTATION", "path": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "sha256": digest(ARCHIVE), "members": len(members)+1, "indexed_members": len(members), "CRC": "PASS", "duplicates": 0, "index_size_hash": "PASS", "fresh_extraction_verifier": recomputed, "byte_identical_recomputed_result_files": result_files, "new_candidates": 0, "new_accounts": 0, "external_GPT_review": "NOT_PERFORMED"}
    write_json(ARCHIVE.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整理并打包货币预期差研究")
    parser.add_argument("action", choices=["prepare", "package"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else package()
