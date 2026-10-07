"""冻结三类连续形态研究协议、直接输入和本轮授权变化。"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.sequential_patterns_regime_v1 import STUDY, digest, now, save_json

OUT = ROOT / "reports/research/510300_sequential_patterns_regime_v1"

PROTOCOL = """# 三类连续形态与状态启停：V1 冻结协议

研究问题是先发生价格过程，再由当时可见状态和已完成案例决定是否启用；检验确认后可执行的剩余收益及启停增量。单一形态不要求十年或逐年赚钱。用户最新验收是完整账户扣费后复合年化至少10%、日收益夏普至少1.2；沿用20万元、最大回撤10%的风险目标、仅510300.SH与CASH_CNY。用户已明确取消年度交易次数下限。

## 一、范围与时间

仅使用已有本地日线和分红资料，不恢复行情采集。本轮是日线1至5个交易日量级研究，不声称验证几十分钟的日内规律。价量表示均为代理，不命名为真实资金净流入或特定行为者意图。

全部本地3479日历史至2026-09-16已经被项目多轮研究使用。本轮按时间顺序重放，但不能称为未见样本、独立样本或事前实际发布的预测。实际协议冻结日期晚于全部行情。历史数据接收时间保留在原表；历史假设的日线可用时点为各交易日收盘之后。新的前向观察必须晚于本轮真实冻结时点，并有当日完整来源记录。

主评价期固定为文件末日向前504个交易日；从该期第一日以20万元重新记账，指标和资格可使用更早已经成熟的案例。这是模拟在该日期启动机制，不是截取长账户中最好的一段。2020-01-02起长重放、2015-01-05至2019-12-31较早失败诊断分别独立记账。不要求这些年份全部盈利，也不允许事后改变主评价期。

## 二、除息中性价格与市场状态

累计价格指数只向前累计已发生分红：前一期指数乘以(当期原始价格+当期每份股息)/前一期原始收盘价；开高低收使用同一前序比例。相对历史边界在该指数坐标中固定。这样除息不自动产生破位。账户使用未复权可成交价，除息日确认前日持有份额的应收股息，到账日才能成为可买入现金。分红记录日应是除息前一交易日，须核对。

RV5、RV20为5/20日收盘对数收益样本标准差；ATR20为20日真实波幅均值。趋势Z=20日对数价格变化/(RV20×sqrt(20))，Z>1为UP，Z<-1为DOWN，其余RANGE。状态用当日收盘已知值。高波动标记为RV20超过此前252日80%分位，只作描述。成交量比=当日成交量/此前20日成交量中位数；下跌幅度/成交量比只作承压代理。每个功能一项主指标，不搜索参数邻域。

## 三、完整过程状态机

每类最多一个在途过程。每个过程从SETUP开始逐日保存，终止后冷却5个交易日；不重复命名连续同一段。前三个252日作为指标预热。

|形态|准备条件与事前边界|后续确认|未确认终止与持有失效|
|---|---|---|---|
|BREAKOUT 收缩后突破|RV5<=0.65 RV20，最近10日高低区间<=最近20日高低区间的70%；锁定当时10日上下沿与ATR|随后最多5日，收盘高于上沿+0.1 ATR|收盘低于下沿-0.1 ATR则向下失败；到期未确认则超时。买入后收盘回到原上沿下方失效|
|RECLAIM 破低后收复|当日收盘低于此前20日最低价-0.1×前日ATR；锁定前低、当日最低价与ATR|随后最多3日，收盘高于原前低+0.1 ATR|收盘低于准备日最低价-1 ATR则继续下跌失败；超时保留。持有后收盘低于准备日最低价-0.1 ATR失效|
|REPAIR 急跌后修复|3日跌幅的对数值<=-2×三日前RV20×sqrt(3)；锁定急跌前收盘、准备日最低价和ATR|随后最多3日，收盘收复从急跌前收盘到迄今最低价的一半跌幅且高于前收盘|收盘低于准备日最低价-1 ATR则继续下跌失败；超时保留。持有后收盘低于确认时已知最低价-0.1 ATR失效|

不允许准备和确认发生在同一天。日内触及与收盘突破仅在日线允许的时间顺序上陈述，不推断单根K线内高低点先后。突破路径另外标记更早交易日试探前低、抬高低点、直接突破，不把试探前低作为全部突破的必要条件。试探未突破、破低未收复、急跌继续下行以及数据终点尚未成熟均完整保存。

## 四、执行与完整账户

收盘确认，下一交易日开盘尝试买入；次日开盘已低于形态失效水平则取消，不重试该入场。每次使用可用现金，100份整手，不融资。多形态同日冲突顺序固定为BREAKOUT、RECLAIM、REPAIR；资金被占用时记录跳过，不能拼接三个账户。持有满5个交易日收盘决定退出，下一开盘卖出。形态失效或近期/完整机制停用同样在下一可卖开盘退出；入场当天禁止卖出。跌停/无成交量不能卖时顺延，不能假设按止损价成交。开盘触及涨停则保守不买。

基础单边佣金万二、压力万四，均最低5元；基础/压力单边滑点5/10基点，价格按0.001元朝不利方向取整。现金利息和无风险收益均按0。终点保持实际持仓收盘估值，另扣卖出费用与滑点储备，不假装已知样本终点提前清仓。最长持有仅为正常可卖情况下5日，连续跌停可延长。记录应收、到账、现金、份额、费用、成交及所有空仓日。

10%为回撤验收线，V1不设置事后反复重置的净值止损。形态退出与启停也不能保证跳空时最大回撤不超线。基础与压力情景完整独立运行。

## 五、近期学习与固定四组比较

独立假设事件记录所有形态确认，即使实际机制停用也继续计算到期结果；用于判断恢复资格。标签沿用形态失效与5日退出，只在退出开盘发生后成为成熟标签，不把在途结果提前加入。每个事件先按20万元固定本金计入压力成本，单次净收益按实际买入资金含佣金作分母。

普通日期对照为相同市场状态下、当时此前504日内已经结束的固定5日买入持有事件，依时间顺序剔除重叠后取最后20个，至少10个。事件发生时锁定对照均值，之后得到该事件的超额标签。它是条件下的普通买入对照，不是独立随机试验。

启停每日仅看此前504日内同形态已完成、依时间顺序剔除重叠后最后12个事件。至少6个且对照完整。近期启用条件：平均压力净收益减去1.2815515655个均值标准误大于0，且平均对照超额大于0。这是正态近似的90%单侧启用启发式，并非多重检验校正后的统计证明。

|固定方案|入场资格|额外退出|
|---|---|---|
|PATTERN_ONLY|所有确认形态|仅形态失效与期限|
|STATE_ONLY|BREAKOUT仅UP；RECLAIM仅RANGE；REPAIR为DOWN或RANGE|仅形态失效与期限；状态只限制入场|
|RECENT_ONLY|近期启用条件|该条件失效时下一开盘退出|
|FULL|固定市场状态、近期启用条件同时成立；上述近期池中同当前状态至少4个且平均净收益与超额均>0|任一资格失效下一开盘退出|

上述状态映射是待验证的机制假设，不是从此次历史结果选出的最优映射。事件过少会导致长期未启用，必须如实报告；不得看到失败后减样本门槛、改回看期、方向或持有期。停用期间继续记录虚拟事件，仅用于研究标签，不连接模拟交易服务。

买入持有与现金为账户基准。再报告同实际平均暴露/同实际波动的理想缩放基准，仅作事后归因，不算真实执行账户或预测。主要近期压力账户用20日联合区块、固定种子20260924、2000次抽样估计FULL相对三个固定比较的年化与夏普差异区间；保存抽样索引。它不能清除跨研究历史污染或模型选择偏差。

## 六、验收、证据与停用

同时报告两档成本下20万元近期完整账户的CAGR>=10%、Sharpe>=1.2、最大回撤<=10%，并检验FULL相对PATTERN_ONLY是否存在启停净增量及是否仅因缩小暴露。任一数值未达标则本版不晋升；全部数值达标也仍需独立前向记录。胜率和单笔盈亏比不能替代全账户指标。没有强制每年交易次数。

本轮不把历史失败自动提升为“形态普遍无效”，也不把任何优势月份拼为新曲线。历史V1结果冻结；前向观察可以继续记录预设规则及未成交、过期和启停理由，不能修改V1协议。若要改变表示，需要独立版本与明确研究假设，不为补救旧版本调参。

最新本地行情早于当前日期时写NO_VIEW/ABSTAIN/POSITION_UNSET，不输出今天的交易判断，也不把它解释为现金仓位。只允许研究观察，不恢复被终止旧策略。现有采集暂停保持；每日任务若设置，只读取已有人为更新的本地输入，不下载行情。无数据变化静默，触发变化、成熟结果或运行失败才记需要关注项。
"""

USER_REQUEST = """# 用户请求及后续澄清

用户于2026-09-24以/goal要求把主线改为“识别当前市场状态下有效的短期交易形态，并及时停用失效形态”，不必每一种规律最近十年持续赚钱。关注价格变化顺序和条件：收缩、试探边界、突破或失败、资金继续推动或撤退。

第一轮限定三类：低波动整理后突破；跌破前低后重新收复；短时急跌后的修复。研究完整连续过程，保留收缩后不突破、破低后不收复等失败。直接突破、抬高低点、破低收复分别统计，不以最终上涨反向定义成功。

资金机制必须映射到可观察行为：破位后能否继续推进、相近成交量下跌幅是否减弱、收复后能否维持。没有订单簿也先研究价量，不能称为真实主力净流入。文献是动机，不能证明今天510300有效。

同时检验形态优势与启停能力。使用当时已知信息决定启停，完整选择执行机制对比始终使用，不拼接各自最好月份。近期数据负责适应，较早数据认识失败，之后数据检验选择；固定窗口和版本，不能亏损后寻找最好片段。考虑识别和执行迟滞以及稀少事件证据不足。

每次触发保存当时状态、形态阶段、入场依据、失效条件和后续结果。比较形态本身与市场状态增量、确认后可成交剩余收益、完整账户改善。执行适配510300 T+1。整套机制扣费后目标年化10%、夏普1.2。规则和数据源固定后，适合每日收盘观察任务。

后续问答原文：
- 问：仓库中还保留“每个完整自然年至少5次交易”的旧要求。这条在新主线中如何处理？
- 答：取消次数下限，以有效机会和整体指标为准。

本文件是用户长消息的忠实结构摘要，不冒充逐字全文。原始长消息保留在任务对话中；以下链接由用户提供：
- https://www.nber.org/papers/w7613
- https://www.fidelity.com/viewpoints/active-investor/bollinger-bands
- https://www.newyorkfed.org/research/staff_reports/sr150.html
- https://doi.org/10.48550/arXiv.1011.6402
- https://doi.org/10.2139/ssrn.2460551
- https://www.bollingerbands.com/bollinger-band-rules
- https://big5.sse.com.cn/site/cht/www.sse.com.cn/lawandrules/guide/jjznlc/c/c_20200722_3986154.shtml
"""


def main():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("协议已经冻结，禁止覆盖。")
    for folder in ("inputs", "code", "results", "sources", "forward"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    src = ROOT / "reports/research/510300_fixed_daily_continuation_v1/2026-09-16"
    mapping = {
        "inputs/prices.parquet": src / "candidate_prices.parquet",
        "inputs/dividends.csv": ROOT / "data/reference/510300_dividends.csv",
        "inputs/dividend_coverage.json": src / "candidate_dividend_coverage.json",
        "inputs/price_admission.json": src / "admission_receipt.json",
        "inputs/calendar.csv": ROOT / "data/reference/sse_trade_calendar_2026.csv",
        "inputs/calendar_metadata.json": ROOT / "data/reference/sse_trade_calendar_2026.metadata.json",
        "inputs/historical_calendar.csv": ROOT / "data/reference/a_share_hs_trading_calendar_2010_2026_v1.csv",
        "inputs/prior_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "inputs/collector_pause.json": ROOT / "reports/research/510300_forward_host_diagnosis_20260922/user_pause_receipt.json",
        "sources/prior_compression_protocol.md": ROOT / "docs/510300_VOLATILITY_COMPRESSION_CHANGE_POINT_V1_SPEC.md",
        "sources/prior_regime_selection_protocol.md": ROOT / "docs/510300_SEQUENTIAL_REGIME_STRATEGY_SELECTION_RULES_20260913.md",
        "code/sequential_patterns_regime_v1.py": ROOT / "research/sequential_patterns_regime_v1.py",
        "code/prepare_sequential_patterns_regime_v1.py": Path(__file__),
        "code/test_sequential_patterns_regime_v1.py": ROOT / "tests/test_sequential_patterns_regime_v1.py",
    }
    for name in ("sina_attempt3.json", "sina_attempt3.raw", "tencent_attempt3.json", "tencent_attempt3.raw",
                 "manager_attempt3.json", "manager_attempt3.raw", "sse_page_1_attempt3.json", "sse_page_1_attempt3.raw", "source_precision_comparison.csv"):
        mapping["sources/recent_price/" + name] = src / name
    coverage = json.loads((src / "candidate_dividend_coverage.json").read_text(encoding="utf-8"))
    for record in coverage["official_source_snapshots"]:
        source = ROOT / record["saved_file"]
        assert digest(source) == record["sha256"]
        mapping["sources/dividends/" + source.name] = source
    for relative, source in mapping.items():
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (OUT / "protocol.md").write_text(PROTOCOL, encoding="utf-8")
    (OUT / "user_request.md").write_text(USER_REQUEST, encoding="utf-8")
    market = pd.read_parquet(OUT / "inputs/prices.parquet")
    dates = pd.to_datetime(market.date).dt.strftime("%Y-%m-%d")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    assert digest(OUT / "inputs/dividends.csv") == coverage["distribution_file_sha256"]
    assert coverage["complete_history_confirmed"] and coverage["coverage_end"] >= dates.iloc[-1]
    for row in div.itertuples():
        if row.ex_date in set(dates):
            ix = dates[dates == row.ex_date].index[0]
            assert dates.iloc[ix - 1] == row.record_date, "分红登记与除息时钟需显式修正，不能跳过。"
    protocol = {"study_id": STUDY, "version": "1.0.0", "frozen_at": now(),
                "capital_cny": 200000, "target_net_cagr": .10, "target_net_sharpe": 1.2,
                "target_max_drawdown": .10, "annual_minimum_cycles": None,
                "primary_start": dates.iloc[-504], "primary_end": dates.iloc[-1],
                "shape_windows": {"compression": [5, 10, 20], "reclaim_reference": 20, "repair": 3},
                "adaptation_lookback_sessions": 504, "max_events": 12, "min_events": 6,
                "min_current_state_events": 4, "max_holding_sessions_before_exit_open": 5,
                "code_is_executable_protocol": "code/sequential_patterns_regime_v1.py",
                "evidence_class": "RETROSPECTIVE_PREQUENTIAL_REPLAY_NOT_UNSEEN_OOS",
                "new_market_collection": False, "orders_authorized": False,
                "historical_failure_does_not_prove_universal_invalidity": True}
    save_json(OUT / "protocol.json", protocol)
    mandate = json.loads((OUT / "inputs/prior_mandate.json").read_text(encoding="utf-8"))
    mandate.update(as_of_date="2026-09-24", latest_user_instruction="以三类连续短期形态和当前市场状态启停为研究主线；整体年化10%、夏普1.2；取消年度次数下限。原不用采集保持。",
                   target_net_sharpe=1.2, target_net_cagr=.10,
                   minimum_complete_cycles_each_full_calendar_year=None, opportunities_per_year_not_quota=True,
                   annual_frequency_requirement_stage="取消最低次数，只披露逐年完整周期。",
                   research_sequence=["固定三类连续过程", "检验确认后可执行收益", "按已成熟事件与当时状态启停", "完整账户比较和前向观察"],
                   current_round=STUDY, current_protocol=str(OUT.relative_to(ROOT) / "protocol.json"),
                   latest_progress_receipt=str(OUT.relative_to(ROOT) / "authority_update.json"),
                   last_research_result="新主线协议固定，结果尚未运行。", goal_achieved=False)
    receipt = {"recorded_at": now(), "user_authorized_new_mainline": True,
               "supersedes_sharpe": {"old": 1.3, "new": 1.2}, "adds_cagr_target": .10,
               "user_cancelled_annual_minimum": True, "minimum_cycles": None,
               "unchanged_collection_pause": True, "new_model_family_not_old_terminal_strategy": True,
               "old_results_and_terminal_rejections_preserved": True,
               "prior_mandate_sha256": digest(OUT / "inputs/prior_mandate.json"),
               "current_mandate": mandate}
    save_json(OUT / "authority_update.json", receipt)
    save_json(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    save_json(ROOT / "config/510300_sequential_patterns_regime_v1.json", protocol)
    save_json(OUT / "source_manifest.json", {name: {"original_path": str(path), "sha256": digest(OUT / name)} for name, path in mapping.items()})
    save_json(OUT / "data_admission.json", {"checked_at": now(), "rows": len(market), "start": dates.iloc[0], "end": dates.iloc[-1],
              "dividend_events": len(div), "record_ex_date_alignment": "PASS", "coverage_hash": "PASS",
              "historical_receipts_are_later_than_observations": True,
              "earlier_price_source_reused_from_saved_admitted_table": True,
              "historical_raw_download_full_rebuild": "NOT_PERFORMED",
              "current_view": "NO_VIEW_STALE_LOCAL_DATA"})
    (OUT / "sources/literature_and_rules.md").write_text("""# 研究动机与执行来源

本地浏览日期2026-09-24。来源只能支持研究动机和交易规则，不能证明510300当前有优势。

- Lo、Mamaysky、Wang，Foundations of Technical Analysis，NBER w7613：https://www.nber.org/papers/w7613 。部分形态具有收益分布信息，不等同于本研究形态有效。该网页访问曾出现间歇错误。
- Osler，Stop-Loss Orders and Price Cascades in Currency Markets：https://www.newyorkfed.org/research/staff_reports/sr150.html 。研究外汇中的止损触发与价格加速，不能直接迁移A股。
- Cont、Kukanov、Stoikov，The Price Impact of Order Book Events：https://arxiv.org/abs/1011.6402 。订单流不平衡与短时价格变化；本轮没有真实订单簿。
- Lo，Adaptive Markets Hypothesis：https://www.mit.edu/~alo/Papers/JPM2004.pdf 。环境变化的理论动机，不是机制已盈利的证据。
- Bollinger规则：https://www.bollingerbands.com/bollinger-band-rules 。关联指标不是独立确认；本轮压缩后需要后续收盘突破。
- 上交所ETF常见问题：https://www.sse.com.cn/assortment/fund/etf/question/ 。股票ETF实施T+1。用户原链是债券ETF指南，也包含股票ETF对照；本轮引用直接ETF问答。
- 用户提供过拟合论文DOI：https://doi.org/10.2139/ssrn.2460551 。当前工具未成功取得该页正文，不将其作为本地实证结果。
- 每日任务产品文档：https://learn.chatgpt.com/docs/automations?surface=app 。本任务工具列表无automation_update；若使用Windows任务计划程序，应如实标记实际宿主，不能声称已创建Codex原生自动化。

本文件为来源链接和核对摘要，未声称包含全部论文原文。价格计算直接输入和14份股息官方文件在包内。
""", encoding="utf-8")
    hashes = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob("*") if p.is_file()}
    save_json(OUT / "freeze.json", {"frozen_at": now(), "study_id": STUDY, "stage": "BEFORE_FIRST_OUTCOME_AND_ACCOUNT_RUN",
                                   "source_count": len(mapping), "hashes": hashes})
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
