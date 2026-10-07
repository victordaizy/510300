"""整理已完成的FOMC历史计算与原文原因；不增加收益窗口或拟合参数。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from historical_fomc_transmission_v1 import GROUPS, make_figure, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_fomc_transmission_v1"
REPORT = OUT / "历史发现_FOMC预期路径与A股剩余收益.md"
SOURCE_PAGE = "https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/"
NEXT_QUESTION = (
    "追查2024至2025年中国股票融资工具从政策宣布到实际使用的历史传导，"
    "区分融资资格、申报额度、实际资金和新增股票购买；先确认是否有可量化的信息增量，"
    "不把本轮美联储分组加条件重调为买入信号。"
)


def local_link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def pct(value):
    return f"{value * 100:+.2f}%"


def collect_coevents():
    targets = [
        {
            "key": "20220316_金融委",
            "url": "https://www.csrc.gov.cn/csrc/c100028/c2093820/content.shtml",
            "published": "2022-03-16，原页面未提供已确认的精确时分",
            "role": "3月17日A股收益的同期国内政策背景，不能将全部缺口和后续收益归因于FOMC。",
            "required_text": "金融稳定发展委员会",
        },
        {
            "key": "20251029_中美元首会晤预告",
            "url": "https://www.fmprc.gov.cn/zyxw/202510/t20251029_11743077.shtml",
            "published": "2025-10-29 15:00，网页标注的北京时间",
            "role": "FOMC之前已有会晤安排，10月30日入场时不是未知事件日。",
            "required_text": "10月30日",
        },
        {
            "key": "20251030_中美元首会晤",
            "url": "https://www.fmprc.gov.cn/wjb_673085/zzjg_673183/xws_674681/xgxw_674683/202510/t20251030_11743847.shtml",
            "published": "2025-10-30 14:22，网页标注的北京时间",
            "role": "ETF入场当天另有中美会晤消息，属于窗口内竞争解释，不倒填至开盘前。",
            "required_text": "14:22",
        },
    ]
    receipt_path = OUT / "sources/coevent_receipts.json"
    path = OUT / "sources/coevents_web_read.json"
    payload = path.read_bytes()
    capture = json.loads(payload.decode("utf-8"))
    captured_text = json.dumps(capture["sources"], ensure_ascii=False)
    receipts = []
    for item in targets:
        if item["url"] not in captured_text or item["required_text"] not in captured_text:
            raise ValueError(f"已保存的网页读取未覆盖同期事件：{item['key']}")
        receipts.append({**item, "path": path.relative_to(ROOT).as_posix(),
                         "retrieved_at": capture["retrieved_at"],
                         "capture_type": "WEB_TOOL_EXTRACT_NOT_ORIGINAL_HTML",
                         "direct_download_note": "证监会直接请求遇到TLS EOF，改存已成功取得的官方网页工具读取，不重复请求。",
                         "sha256": hashlib.sha256(payload).hexdigest()})
    save(receipt_path, receipts)
    return receipts


def reviewed_facts():
    facts = [
        {
            "date": "2022-03-16", "statement_action": "加息25个基点至0.25%—0.50%。",
            "upstream_facts": "高通胀、极紧劳动力市场与强劲需求构成收紧理由；俄乌冲突同时带来物价上行与增长下行风险。",
            "communication_facts": "鲍威尔表示需求、就业及居民企业资产负债表较强，认为经济可以承受收紧，短期衰退风险并不高。",
            "transcript_pages": [1, 4, 5],
            "interpretation": "加息、未来收紧安排与经济韧性信息并存。声明窗口股跌，记者会股涨；合并后利率升股涨与混合信息相容，不能识别成纯增长好消息。",
            "competing_explanation": "中国3月16日金融委会议已有资本市场政策信号，3月17日ETF缺口和收益不是美国会议的纯效果。",
        },
        {
            "date": "2023-12-13", "statement_action": "利率维持5.25%—5.50%，当次没有降息。",
            "upstream_facts": "近期通胀读数回落、经济活动较第三季度放缓，限制性政策已在压低需求与通胀。",
            "communication_facts": "政策利率可能处于或接近本轮峰值；SEP参与者对2024年末合适利率的中位判断为4.6%，同时强调这不是委员会承诺。",
            "transcript_pages": [3, 4],
            "interpretation": "本次未降息而1年OIS明显下行，与未来路径向宽松修正相容；不是实际降息幅度本身。10年名义利率的下行伴随实际利率更大下行，不能解释为纯通胀预期下降。",
            "competing_explanation": "美国折现条件变化能否压过中国本地盈利、风险偏好和资金约束，不能仅凭美国股债同涨推定。",
        },
        {
            "date": "2024-07-31", "statement_action": "利率维持5.25%—5.50%。",
            "upstream_facts": "第二季度通胀进展增加信心，决策同时权衡过早降息使通胀反复与过晚降息损害就业活动。",
            "communication_facts": "鲍威尔表示若后续整体数据符合要求，9月可能考虑降息，但当时尚未决定。",
            "transcript_pages": [3, 4],
            "interpretation": "1年OIS下行而标普仅跌0.011692%，接近不变。负号不足以把该事件命名为已识别的增长恐慌；其后A股下跌也不能反过来确认这个标签。",
            "competing_explanation": "有限窗口、接近零的股票变化、后续国内外其他消息均限制四象限原因识别。保留原分组，不看收益后设置新的忽略区间。",
        },
        {
            "date": "2025-10-29", "statement_action": "降息25个基点至3.75%—4.00%，并决定12月1日起停止缩表。",
            "upstream_facts": "就业下行风险上升促成本次降息；通胀上行风险与就业下行风险并存，未来政策存在分歧。回购相对管理利率走高、常备回购使用增加、有效联邦基金利率向准备金利率靠近，是停止缩表的运行依据。",
            "communication_facts": "鲍威尔明确表示12月再降息远非已成定局，且官员对此意见分歧。停止缩表旨在维持充裕准备金条件，不等于保证下一次再降息。",
            "transcript_pages": [2, 3, 5, 7, 9],
            "interpretation": "实际降息与额外降息预期减弱可以同时出现。1年OIS上升主要发生在记者会窗口，与后续路径重新定价相容；不能把上升直接解释成当次加息或纯通胀恶化。",
            "competing_explanation": "ETF入场当天另有已预告的中美元首会晤，5日和20日收益不是FOMC的独立因果效果。",
        },
    ]
    for item in facts:
        stamp = item["date"].replace("-", "")
        item["statement_url"] = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm"
        item["transcript_url"] = f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{stamp}.pdf"
        item["statement_local"] = f"sources/{stamp}_声明.html"
        item["transcript_local"] = f"sources/{stamp}_记者会.pdf"
        item["fact_and_interpretation_separated"] = True
    return facts


def build_report(events, summaries, result, facts):
    table = pd.DataFrame(summaries)
    lines = [
        "# 历史发现：FOMC预期路径与A股剩余收益", "",
        "2026年9月30日。仅分析2022—2025年已经发生的32次预定会议，沿用既有510300行情、分红与压力费用；没有预测任务。", "",
        "**本轮查明了实际政策动作、市场重估的未来利率路径、停止缩表的运行原因可以不同；同时没有找到这些美国消息组合在510300下一可交易开盘之后的稳定买入优势。夏普1.2目标尚未实现。**", "",
        "样本覆盖四年，是为比较两个固定历史阶段，未要求因子经过五年或十年同方向检验。事件取全，不依A股涨跌选会议；4个原文案例按各组1年OIS绝对变化最大选定，选择时没有使用本轮ETF收益。历史行情此前已见过，本轮不是独立验证。", "",
        "**先把被研究的变化说清楚。** 1年OIS是未来短期利率路径相关的市场价格，不是美联储当次加减息数值，也不能完全排除风险溢价。与它同窗的标普500正负号只提供混合消息的线索。数据来自旧金山联储USMPD当前历史版本，使用原始变化字段，没有引入全样本拟合的主成分或正交冲击。数据库与原始报价不同：本次不能证明当年的首次公开数据库版本或实时数据访问能力。[数据库及下载说明](" + SOURCE_PAGE + ")。", "",
        "窗口采用声明前10分钟至记者会开始后60分钟，样本均为100分钟；声明30分钟、记者会70分钟分别保留。这是固定观察窗口，不等于包含整场记者会的每句话。按照美国东部夏令时转换为上海时间，收益起点为窗口结束后首个实际A股开盘。[数据库论文的窗口与单位说明](https://www.frbsf.org/wp-content/uploads/wp2025-30.pdf)。", "",
        f"32次中有**{result['sample']['statement_full_rate_sign_reversals']}次**，声明窗口与声明加记者会窗口的OIS变化方向相反；四种股债组合有**{result['sample']['statement_full_group_changes']}次**发生改变。因此只把政策声明的第一反应视为整场信息，会改变本轮相当一部分原因分类。", "",
        "**四个原文案例：事实、解释与竞争原因分列。**", "",
        "| 美国会议日期 | 实际政策动作 | 1年OIS变化 | 同窗标普500 | 510300入场缺口 | 入场后5日净收益 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selected = events[events.cause_case_selected].set_index(events.loc[events.cause_case_selected, "us_date"].dt.strftime("%Y-%m-%d"))
    for fact in facts:
        row = selected.loc[fact["date"]]
        action = fact["statement_action"].split("，")[0]
        lines.append(f"| {fact['date']} | {action} | {row.OIS1Y_bp:+.3f}基点 | {row.SP500_pct:+.3f}% | {pct(row.gap_total_return)} | {pct(row.net_return5)} |")
    lines += ["", "入场缺口是上次A股收盘到本次开盘的含息变化，混合其间全部消息，不能归因于单个FOMC事件，也没有计入入场后的盈利。", ""]
    for fact in facts:
        row = selected.loc[fact["date"]]
        pages = "、".join(str(page) for page in fact["transcript_pages"])
        lines += [
            f"**{fact['date']}。** {fact['upstream_facts']}{fact['communication_facts']}[当次声明]({fact['statement_url']})、[记者会第{pages}页]({fact['transcript_url']})。", "",
            f"本轮解释：{fact['interpretation']} 声明窗口OIS变化{row.statement_OIS1Y_bp:+.3f}基点，记者会窗口{row.pressconf_OIS1Y_bp:+.3f}基点。竞争解释：{fact['competing_explanation']}", "",
        ]
    lines += [
        "中国同期消息可由原文定位：2022年3月16日金融委会议强调市场稳定、政策协调及积极应对；页面只有日期，未据此编造精确发布分钟。2025年10月29日15:00已预告次日中美元首会晤，30日14:22又有会晤新闻，后者晚于ETF入场开盘。[金融委会议](https://www.csrc.gov.cn/csrc/c100028/c2093820/content.shtml)、[会晤预告](https://www.fmprc.gov.cn/zyxw/202510/t20251029_11743077.shtml)、[会晤新闻](https://www.fmprc.gov.cn/wjb_673085/zzjg_673183/xws_674681/xgxw_674683/202510/t20251030_11743847.shtml)。", "",
        "**10年收益率不能只看总数。** 下表是同窗名义收益率、TIPS实际收益率与两者差额的算术分解，单位均为基点。通胀补偿包含通胀风险和相对流动性等溢价，TIPS实际收益率也不是纯粹的未来实际无风险利率。本轮没有独立估计期限、违约或流动性补偿，不能把这些分量当成已识别原因。", "",
        "| 会议日期 | 10年名义收益率变化 | 10年实际收益率变化 | 两者差额：通胀补偿变化 |",
        "|---|---:|---:|---:|",
    ]
    for date, row in selected.iterrows():
        lines.append(f"| {date} | {row.UST10Y_bp:+.3f} | {row.TIPS10Y_bp:+.3f} | {row.inflation_compensation_10y_bp:+.3f} |")
    lines += [
        "", "2023年12月名义利率下降时通胀补偿反而上升，2025年10月名义利率上升时通胀补偿略降。这两个具体历史例子足以否定‘名义利率变化就是通胀预期同向变化’的简单归因，但不能据此宣称完整分离了全部风险溢价。", "",
        "**开盘后还剩多少收益。** 以下均为每次10万元独立事件预算的投入金额净收益，佣金每边0.04%、最低5元，滑点每边0.10%，按0.001元报价档位和100份整手计算，并计入实际持有期间的分红权益。它不是20万元完整账户收益；没有将重叠事件拼成可执行持仓。5个交易日是事先确定的主窗口，20日只提供较长传导背景。", "",
        "| 同窗组合 | 次数 | 5日净收益均值 | 5日中位数 | 正收益次数 | 去掉最好一次后的均值 | 20日净收益均值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in GROUPS[:4]:
        short = table[(table.phase == "全部") & (table.grouping == "group") & (table.label == label) & (table.horizon == 5)].iloc[0]
        longer = table[(table.phase == "全部") & (table.grouping == "group") & (table.label == label) & (table.horizon == 20)].iloc[0]
        lines.append(f"| {label} | {int(short.n)} | {pct(short['mean'])} | {pct(short['median'])} | {int(short.positive)}/{int(short.n)} | {pct(short.mean_without_best)} | {pct(longer['mean'])} |")
    lines += [
        "", "唯一5日均值略正的‘利率降股跌’只有4次，去掉最好一次即为负值，且原文案例已经说明负号可能接近零。这里展示删去最好一次只是说明均值集中程度，不是删样本后的新策略。所有原事件仍留在计算中，不反向交易、不改分组阈值。", "",
        "固定历史阶段的5日均值如下；样本少意味着只能描述这些已发生事件，不把正数称为稳定规律。", "",
        "| 同窗组合 | 2022—2023年：次数、均值 | 2024—2025年：次数、均值 |",
        "|---|---:|---:|",
    ]
    for label in GROUPS[:4]:
        values = []
        for phase in ["2022-2023", "2024-2025"]:
            stat = table[(table.phase == phase) & (table.grouping == "group") & (table.label == label) & (table.horizon == 5)].iloc[0]
            values.append(f"{int(stat.n)}次；{pct(stat['mean'])}")
        lines.append(f"| {label} | {' | '.join(values)} |")
    lower_rate = table[(table.phase == "全部") & (table.grouping == "rate_group") & (table.label == "利率下降") & (table.horizon == 5)].iloc[0]
    higher_rate = table[(table.phase == "全部") & (table.grouping == "rate_group") & (table.label == "利率上升") & (table.horizon == 5)].iloc[0]
    favorable20 = table[(table.phase == "全部") & (table.grouping == "group") & (table.label == "利率降股涨") & (table.horizon == 20)].iloc[0]
    september = events[events.us_date.eq(pd.Timestamp("2024-09-18"))].iloc[0]
    lines += [
        "", f"若只按利率变化分组，下降的{int(lower_rate.n)}次5日均值{pct(lower_rate['mean'])}，上升的{int(higher_rate.n)}次为{pct(higher_rate['mean'])}。加入美国股票的方向后，本轮没有出现足以支持买入的清楚分离。", "",
        f"‘利率降股涨’的20日均值为{pct(favorable20['mean'])}，去掉最好一次后为{pct(favorable20.mean_without_best)}。该最好事件是2024年9月18日FOMC：9月19日开盘入场后5日{pct(september.net_return5)}、20日{pct(september.net_return20)}。这些持有窗口包含9月24日中国股票融资工具等政策的宣布，不能把随后整段A股上涨都归功于美国利率路径。9月24日原文明确介绍互换便利、回购增持再贷款等工具。[中国政策发布会](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)。", "",
        "2024年5月1日及2025年1月29日会议分别直到5月6日、2月5日才有下一个A股开盘，等待102小时和149小时。这两条照常保留；跨假期期间其他消息更多，不能假设美股反应在A股开盘仍是唯一新增信息。", "",
        "**本轮留下的实用判断。** 判断外部利率因子，至少要同时知道当次动作、后续路径沟通和政策面临的具体约束；判断A股机会还需要中国自身传导渠道和入场后剩余空间。2025年10月的准备金运行原因与利率路径分歧可以并存，说明把所有‘宽松’压成一个利好标签会丢失关键信息。这个判断有原文与同窗价格支持；可持续的510300净收益优势尚未建立。", "",
        "本轮只核对了源字段与窗口对应、开盘在消息之后、5/20日端点、分红与费用计算恒等，以及图表可读性。实际开盘价用于预算内整手数量示例，未证明开盘前可按该精确数量提交成交订单。没有新增策略账户、拟合参数或前瞻卡，也没有宣称夏普达标。", "",
        "下一历史问题：" + NEXT_QUESTION, "",
        local_link(OUT / "FOMC原因线索与A股剩余收益.png", "原因组合与两阶段收益图") + "；" + local_link(OUT / "32次FOMC原因线索与开盘后收益.csv", "全部32次事件明细") + "；" + local_link(OUT / "reviewed_case_facts.json", "四例原文事实与解释") + "；" + local_link(OUT / "sources/case_receipts.json", "美联储原文索引") + "。", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def update_status(result):
    relative = lambda p: p.relative_to(ROOT).as_posix()
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(path.read_text(encoding="utf-8"))
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": relative(OUT / "result.json"),
        "last_research_result": "32次FOMC中8次声明与合并窗口的利率变化反号；2025年10月实际降息而未来路径重新向上定价。利率降股涨14次，510300下一开盘后5日净均值-0.77%；20日正均值依赖2024年9月中国政策同期上涨。完成原因解释，未建立独立ETF买入优势。",
        "last_source_result": "保存USMPD当前历史工作簿、4份原声明、4份记者会和3份同期中国官方消息；行情及分红复用本地，仅研究已发生历史。",
        "latest_historical_diagnostic_at": result["recorded_at"], "latest_historical_report": relative(REPORT),
        "latest_historical_fomc_transmission": relative(OUT / "result.json"),
        "latest_continuation_report": relative(REPORT),
        "latest_continuation_classification": result["classification"],
        "current_driver_continuation_classification": result["classification"],
        "current_driver_consecutive_blocked_goal_turns": 0,
        "next_research_question": NEXT_QUESTION,
        "goal_achieved": False, "goal_status": "active",
    })
    save(path, mandate)
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.update({"latest_completed_study": relative(OUT / "result.json"),
                "latest_report": relative(REPORT), "updated_at": result["recorded_at"]})
    save(path, cfg)
    path = ROOT / "RESEARCH_STATUS.md"
    previous = path.read_text(encoding="utf-8")
    marker = "<!-- HISTORICAL_FOMC_TRANSMISSION_V1_20260930 -->"
    prefix = (
        marker + "\n\n## 2026-09-30 FOMC历史原因已区分，开盘后的ETF买入优势未成立\n\n"
        "2022—2025年全部32次预定会议，按固定1年OIS与标普500同窗正负分组。8次声明与声明加记者会窗口的OIS方向反转，13次组合改变。4例原文显示实际政策动作、未来利率路径及停止缩表的运行原因必须分开；2025年10月实际降息25基点，1年OIS仍上升8.92基点。2024年7月美股仅跌0.012%，不能据负号认定增长恐慌。\n\n"
        "利率降股涨14次，下一A股开盘后5日压力费用净均值-0.77%，4次为正。20日均值+0.49%，去掉最好一次后-1.32%；最好窗口包含2024年9月中国重要政策。其余组合没有稳定优势。四象限不是纯原因识别，当前历史数据库不是当年首版；事件收益不是完整账户。原样本全部保留，不加条件调参。\n\n"
        "新增账户、拟合参数、前瞻卡均0。夏普1.2及年化10%目标仍未实现，继续历史研究。下一问题是中国股票融资工具从宣布到实际使用的历史传导。\n\n"
        "详见" + local_link(REPORT, "FOMC预期路径与A股剩余收益") + "和" + local_link(OUT / "32次FOMC原因线索与开盘后收益.csv", "32次事件明细") + "。\n\n"
    )
    if marker not in previous:
        path.write_text(prefix + previous, encoding="utf-8")


def main():
    events = pd.read_parquet(OUT / "32次FOMC原因线索与开盘后收益.parquet")
    summaries = json.loads((OUT / "分组描述.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "calculation_result.json").read_text(encoding="utf-8"))
    coevents = collect_coevents()
    facts = reviewed_facts()
    save(OUT / "reviewed_case_facts.json", facts)
    result.update({
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_CAUSES_REVIEWED_NO_ROBUST_OPEN_AFTER_BUY_EDGE",
        "classification": "PROGRESS_HISTORICAL_FOMC_CAUSES_AND_REMAINING_RETURNS",
        "reviewed_fed_statements": 4, "reviewed_fed_press_conferences": 4,
        "reviewed_chinese_coevent_documents": len(coevents),
        "source_vintage": "2026年9月下载的USMPD历史版本；不是当年数据库首版。",
        "case_facts": (OUT / "reviewed_case_facts.json").relative_to(ROOT).as_posix(),
        "supported_findings": [
            "当次政策动作与未来路径重估方向可以相反，2025年10月降息但1年OIS上升。",
            "停止缩表可由准备金运行条件促成，不能与后续政策利率决策混成一个宽松因子。",
            "股债正负组合是原因线索，接近零的股票变化不能识别增长恐慌。",
            "本轮固定组合没有建立510300下一可用开盘后稳定的买入净收益优势。",
        ],
        "not_established": ["纯政策或信息冲击识别", "全部名义利率风险溢价分解", "FOMC对后续A股收益的独立因果效应", "完整账户夏普达标"],
        "old_accounts_unchanged": True, "next_research_question": NEXT_QUESTION,
    })
    build_report(events, summaries, result, facts)
    make_figure(events, summaries)
    save(OUT / "result.json", result)
    update_status(result)
    print(json.dumps({key: result[key] for key in ["study_id", "recorded_at", "status", "classification", "sample", "new_accounts", "goal_achieved", "report"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
