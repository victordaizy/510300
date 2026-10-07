"""全因素联合讨论与解释评分；不训练、不生成交易指令或金融账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_all_factor_joint_scorecard_v1"
TZ = ZoneInfo("Asia/Shanghai")
INPUTS = {
    "daily": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/results/全部3488当时已知技术与宏观_未知保留.parquet",
    "points": "reports/research/510300_core_support_complement_description_v1/implementation_v1_0_1/results/全部12支持接受_公布锚价格时钟与宏观原值.parquet",
    "keys": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/results/事前固定17关键日期_宏观与技术.parquet",
    "credit": "reports/research/510300_policy_expectation_thesis_exit_v1/implementation_v1_0_1/results/全部84月_双期限兑现与缺失不删.parquet",
    "brokers": "reports/research/510300_source_expectation_transmission_review_v1/results/七券商报告_方法用途与历史预测分开.parquet",
    "old_joint": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/summary.json",
    "old_nonlinear": "reports/research/510300_multidim_nonlinear_score_v1/result.json",
    "valuation": "reports/research/510300_historical_index_valuation_repricing_v1/daily_valuation.parquet",
    "earnings": "reports/research/510300_historical_index_earnings_population_v1/aggregates.parquet",
}

# 同源指标都展示，但不把同一个价量、政策公告或资金来源计成多份独立证据。
GROUPS = [
    ("价格结构", "确认", [
        ("原始收盘", "close"), ("现金股息调整价", "ac"), ("支持区域上沿", "setup_high"),
        ("支持区域下沿", "setup_low"), ("EMA偏离/ATR", "ema_distance_atr"),
        ("突破与回踩结构", None), ("日周线支撑阻力", None), ("跳空与价格缺口", None)]),
    ("动量与技术", "确认", [
        ("日线MACD/ATR", "daily_hist_atr"), ("上一已完成周MACD/ATR", "weekly_hist_atr"),
        ("五日标准化动量", "momentum5_scaled"), ("RSI", None), ("KDJ", None),
        ("ADX趋势强度", None), ("背离与趋势衰减", None)]),
    ("量价参与", "确认", [
        ("相对成交量", "relative_volume"), ("五日上涨量占比", "up_volume_balance5"),
        ("日内实体效率", "body_efficiency"), ("成交金额", "amount"),
        ("OBV", None), ("MFI", None), ("真实换手率", None)]),
    ("波动与风险", "风险", [
        ("ATR14相对价格", "atr_percent"), ("短长实现波动比", "rv_ratio"),
        ("原已知ES95", "known_es95"), ("下行波动", None),
        ("止损距离与扣费后目标空间", None), ("账户回撤与集中度", None)]),
    ("宏观周期", "背景", [
        ("PMI新订单水平", "pmi_orders"), ("PMI新订单变化", "pmi_orders_change"),
        ("PMI生产与就业", None), ("企业经营预期", None), ("M1/M2及统计制度", None),
        ("社融及信用组成", None), ("PPI/CPI", None), ("工业生产", None),
        ("消费与出口", None), ("GDP与财政节奏", None)]),
    ("政策与预期兑现", "催化/失效", [
        ("原支持信息与公布上界", "support_information"), ("去重共同来源数", "support_common_count"),
        ("当月LPR双期限兑现", "credit_state"), ("OMO事前数量预期", None),
        ("降准及净流动性增量", None), ("财政政策实施与支出", None),
        ("宣布实施市场接受时序", None)]),
    ("资金价格与融资", "传导", [
        ("DR007与政策利率差", "funding_gap_pp"), ("资金差五日变化", "funding_gap_change5"),
        ("融资余额五日变化", "financing_net_change5"), ("融资买入活跃度", "financing_buy_activity"),
        ("信用利差", None), ("银行贷款与融资需求", None), ("回购计划实施", None)]),
    ("真实流量与订单流", "传导", [
        ("ETF连续新份额变化", None), ("PCF/IOPV与申赎", None), ("ETF溢折价", None),
        ("北向披露可比口径", None), ("逐笔主动买卖与CVD", None),
        ("L2分类资金流定义", None)]),
    ("指数主线与广度", "传导", [
        ("当时指数成员及权重", None), ("行业相对强弱", None), ("上涨成分扩散", None),
        ("强势行业持续性", None), ("主线对510300贡献", None), ("行业盈利与价格接受", None)]),
    ("估值与盈利", "空间/传导", [
        ("指数PE/PB", None), ("股息率", None), ("股债收益差", None),
        ("整体盈利与利润率", None), ("EPS预测及修订", None),
        ("盈利改善广度", None), ("预期已反映的程度", None)]),
    ("博弈与情绪", "路径/风险", [
        ("市场拥挤与持仓披露", None), ("IF基差", None), ("期权IV/PCR/偏斜", None),
        ("涨跌停与极端广度", None), ("新股解禁与资金供给", None), ("制度变化及再平衡", None)]),
    ("外部环境与事件", "背景/风险", [
        ("人民币汇率", None), ("美元与美债利率", None), ("美联储与全球流动性", None),
        ("海外市场已知收盘", None), ("商品与油价", None), ("地缘及突发事件", None)]),
]


def timestamp():
    return datetime.now(TZ).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def catalogue():
    rows = []
    for number, (group, role, factors) in enumerate(GROUPS, 1):
        for index, (name, binding) in enumerate(factors, 1):
            rows.append({"factor_id": f"G{number:02d}_{index:02d}", "group": group,
                         "role": role, "factor": name, "binding": binding,
                         "binding_status": "BOUND_DESCRIPTIVE_ONLY" if binding else "NOT_BOUND_IN_THIS_CARD",
                         "independent_vote": False,
                         "limitation": "绑定只代表本轮解释可读；不代表历史首版认证或预测有效。未绑定不代表项目无此数据。"})
    return rows


def available(value, cutoff):
    if value is None or pd.isna(value):
        return False
    date = pd.Timestamp(value)
    if date.tzinfo is None:
        return False
    return date.tz_convert("Asia/Shanghai") <= cutoff


def number(row, name):
    value = row.get(name)
    return float(value) if value is not None and pd.notna(value) and math.isfinite(float(value)) else None


def credit_at(ledger, cutoff):
    month = cutoff.strftime("%Y-%m")
    rows = ledger[ledger["month"].astype(str).eq(month)]
    if len(rows) != 1:
        return None, "当月完整记录未知"
    row = rows.iloc[0]
    if not available(row.announcement_at, cutoff):
        return None, "当月LPR尚未公布"
    if (row.get("same_instrument_prior_version_clock_valid") != True
            or not available(row.survey_published_at, pd.Timestamp(row.announcement_at))
            or not available(row.survey_modified_at, pd.Timestamp(row.announcement_at))):
        return None, "事前同工具调查或版本钟不具备"
    state = row.get("joint_credit_state")
    if state not in {"DUAL_CREDIT_UNDERDELIVERY", "MATCHED_BOTH_TENORS", "DUAL_CREDIT_OVERDELIVERY"}:
        return None, "两期限共同状态未识别"
    return state, "原调查的公布/修改上界合格；历史首版仍未独立认证"


def observations(row, support, ledger, cutoff):
    values = {key: number(row, key) for key in (
        "close", "ac", "amount", "ema_distance_atr", "daily_hist_atr", "momentum5_scaled",
        "up_volume_balance5", "body_efficiency")}
    rv, vol = number(row, "log_rv_ratio"), number(row, "log_relative_volume")
    values["rv_ratio"] = math.exp(rv) if rv is not None else None
    values["relative_volume"] = math.exp(vol) if vol is not None else None
    atr, close = number(row, "atr14"), number(row, "close")
    values["atr_percent"] = 100. * atr / close if atr is not None and close else None
    weekly_last, weekly_available = row.get("weekly_last_date"), row.get("weekly_available_date")
    weekly_ok = (pd.notna(weekly_last) and pd.notna(weekly_available)
                 and pd.Timestamp(weekly_last).normalize() < cutoff.tz_localize(None).normalize()
                 and pd.Timestamp(weekly_available).normalize() <= cutoff.tz_localize(None).normalize())
    values["weekly_hist_atr"] = number(row, "weekly_hist_atr") if weekly_ok else None
    orders_ok = available(row.get("orders_available_at"), cutoff) and row.get("orders_known") == True
    level = number(row, "pmi_orders_level")
    values["pmi_orders"] = level + 50. if orders_ok and level is not None else None
    values["pmi_orders_change"] = number(row, "pmi_orders_change") if orders_ok else None
    funding_ok = (row.get("funding_known") == True and available(row.get("funding_available_at"), cutoff)
                  and available(row.get("funding_policy_known_at"), cutoff))
    for key in ("funding_gap_pp", "funding_gap_change5"):
        values[key] = number(row, key) if funding_ok else None
    margin_ok = row.get("margin_known") == True and available(row.get("margin_available_at"), cutoff)
    for key in ("financing_net_change5", "financing_buy_activity"):
        values[key] = number(row, key) if margin_ok else None
    support_ok = support is not None and available(support.get("source_available_upper"), cutoff)
    values["support_information"] = True if support_ok else None
    values["support_common_count"] = (len(set(str(support["source_common_ids"]).split("|")))
                                      if support_ok else None)
    for key, source in (("setup_high", "support_setup_high"), ("setup_low", "support_setup_low"),
                        ("known_es95", "known_es95")):
        values[key] = number(support, source) if support_ok else None
    values["credit_state"], credit_reason = credit_at(ledger, cutoff)
    reasons = {"weekly_clock": "只使用已完成且已经可得的周线",
               "macro_clock": "原表16:00快照重新按本轮15:05过滤，不能沿用旧known标记",
               "credit": credit_reason,
               "policy": "只绑定原已记录支持源，不代表全部政策覆盖或同工具超预期"}
    return values, reasons


def core_score(values, review_existing_position=False):
    """工作等级采用条件组合；全因素预测总分留待完整用途检验。"""
    hist, momentum = values.get("daily_hist_atr"), values.get("momentum5_scaled")
    if hist is None or momentum is None:
        return {"core_explanatory_score": None, "score_reason": "价格修复核心输入未知", "liquidity_state": "UNKNOWN"}
    score = 40 if hist > 0 and momentum > 0 else 20 if hist > 0 or momentum > 0 else 0
    anchor_known = values.get("setup_high") is not None and values.get("ac") is not None
    price_accepted = anchor_known and values["ac"] > values["setup_high"]
    if score == 40 and price_accepted and values.get("support_information") is True:
        score = 60
    gap, financing = values.get("funding_gap_pp"), values.get("financing_net_change5")
    liquidity = ("UNKNOWN" if gap is None or financing is None else
                 "BANK_AND_MARGIN_ALIGNED" if gap < 0 and financing > 0 else
                 "BANK_AND_MARGIN_ADVERSE" if gap >= 0 and financing <= 0 else "MIXED")
    if (score == 60 and liquidity == "BANK_AND_MARGIN_ALIGNED"
            and values.get("relative_volume") is not None and values["relative_volume"] > 1
            and values.get("up_volume_balance5") is not None and values["up_volume_balance5"] > 0):
        score = 80
    reason = "核心解释等级：0无日线修复，20部分修复，40价量背景待传导，60支持与价格接受，80再有银行资金/融资/成交参与共同确认"
    if review_existing_position and values.get("credit_state") == "DUAL_CREDIT_UNDERDELIVERY":
        score = min(score, 20)
        reason = "只复查已有支持持仓：同工具LPR调查兑现偏弱，支持论据冲突；不是新开仓分或政策冲击因果"
    weekly, pmi = values.get("weekly_hist_atr"), values.get("pmi_orders")
    context = ("周期未知" if pmi is None else "订单收缩背景" if pmi < 50 else "订单扩张背景")
    context += "；" + ("周线未知" if weekly is None else "周线仍弱/日线修复" if weekly < 0 else "日周动量同向背景")
    return {"core_explanatory_score": score, "score_reason": reason, "liquidity_state": liquidity,
            "context": context, "full_factor_score": "NOT_COMPUTED_INCOMPLETE_BINDINGS_AND_NO_VALIDATED_MODEL",
            "score_is_win_probability": False, "trade_authorized": False}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError("本用途已登记，禁止覆盖协议。")
    sources = []
    for key, rel in INPUTS.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError("本轮来源缺失：" + rel)
        sources.append({"key": key, "path": rel, "sha256": digest(path)})
    protocol = {
        "study_id": "510300_ALL_FACTOR_JOINT_SCORECARD_V1", "registered_at": timestamp(),
        "user_instruction": "所有因素全部加进来讨论，进行打分",
        "purpose": "全因素目录、原完整案例事实矩阵与联合解释工作等级；非预测或金融策略",
        "sources": sources, "code_sha256": digest(Path(__file__).absolute()),
        "catalogue": catalogue(), "cutoff": "历史交易日15:05 Asia/Shanghai",
        "cases": "全部12原支持接受日＋原固定17关键日＋2023-06-20已有持仓兑现复查；去重完整保留",
        "history_role": "DEVELOPMENT_ALREADY_OBSERVED_NOT_INDEPENDENT",
        "grades": "0/20/40/60/80是事前本用途固定解释等级；100要求完整主线/盈利/估值/风险等不同证据，尚未实现",
        "not_allowed": ["按已知输赢改等级", "多指标同源重复加分", "未知填0", "把工作分当胜率",
                        "把原失败模型直接激活", "原金融策略修改", "新金融回测", "新股票收益标签", "实盘委托"],
        "financial_admission": "NOT_ADMITTED", "new_fits": 0, "new_accounts": 0,
        "previous_results": "原联合树、非线性评分及全部旧固定金融裁决保留，不以本目录营救",
    }
    write_json(protocol_path, protocol)
    print("已登记全因素联合解释评分；未准入预测或金融用途。")


def run():
    protocol = read_json(OUT / "protocol.json")
    if protocol["code_sha256"] != digest(Path(__file__).absolute()):
        raise ValueError("登记后代码发生变化，不能执行本协议。")
    for source in protocol["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("来源变化：" + source["path"])
    if (OUT / "run_started.json").exists():
        raise FileExistsError("本轮解释生成已启动，禁止覆盖重跑。")
    daily = pd.read_parquet(ROOT / INPUTS["daily"]).set_index("date", verify_integrity=True)
    points = pd.read_parquet(ROOT / INPUTS["points"])
    keys = pd.read_parquet(ROOT / INPUTS["keys"])
    ledger = pd.read_parquet(ROOT / INPUTS["credit"])
    dates = sorted(set(pd.to_datetime(points.origin)) | set(pd.to_datetime(keys.date)) | {pd.Timestamp("2023-06-20")})
    assert len(points) == 12 and len(keys) == 17 and len(dates) == 30
    write_json(OUT / "run_started.json", {"at": timestamp(), "new_accounts": 0, "new_fits": 0})
    case_rows, factor_rows = [], []
    cat = catalogue()
    for date in dates:
        cutoff = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        match = points[pd.to_datetime(points.origin).eq(date)]
        review = date == pd.Timestamp("2023-06-20")
        if review:
            match = points[pd.to_datetime(points.origin).eq(pd.Timestamp("2023-06-15"))]
        support = match.iloc[0] if len(match) == 1 else None
        values, reasons = observations(daily.loc[date], support, ledger, cutoff)
        score = core_score(values, review)
        observed = 0
        for factor in cat:
            binding = factor["binding"]
            value = values.get(binding) if binding else None
            present = value is not None
            observed += int(present)
            factor_rows.append({**factor, "date": date, "decision_at": cutoff.isoformat(),
                                "value": str(value) if present else None,
                                "observation_status": "OBSERVED_DEVELOPMENT_ONLY" if present else "UNKNOWN_NOT_ZERO",
                                "history_first_vintage": "NOT_CERTIFIED",
                                "numeric_predictive_vote": "NOT_ADMITTED"})
        case_rows.append({"date": date, "decision_at": cutoff.isoformat(),
                          "case_role": "EXISTING_POSITION_THESIS_REVIEW" if review else "ORIGINAL_SUPPORT_ACCEPTANCE" if support is not None else "ORIGINAL_FIXED_KEY_DATE",
                          **values, **score, "observed_factor_slots": observed,
                          "total_factor_slots": len(cat), "observation_coverage_percent": 100. * observed / len(cat),
                          "history_first_vintage": "NOT_CERTIFIED", "scope": "EXPLANATION_NOT_NEW_ENTRY_OR_PREDICTION",
                          "clock_notes": json.dumps(reasons, ensure_ascii=False)})
    results = OUT / "results"
    results.mkdir()
    cases, factors, catalog = pd.DataFrame(case_rows), pd.DataFrame(factor_rows), pd.DataFrame(cat)
    for name, frame in (("全部因素目录_同源角色与未绑定", catalog), ("全部30历史案例_联合解释分与覆盖", cases), ("全部案例逐因素原值_未知不填零", factors)):
        frame.to_parquet(results / (name + ".parquet"), index=False)
        frame.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {"study_id": protocol["study_id"], "completed_at": timestamp(),
               "status": "COMPLETED_ALL_FACTOR_DISCUSSION_AND_PARTIAL_EXPLANATORY_SCORES_NOT_PREDICTIVE",
               "groups": len(GROUPS), "factor_slots": len(cat), "bound_factor_slots": sum(x["binding"] is not None for x in cat),
               "historical_cases": len(cases), "case_factor_rows": len(factors),
               "coverage_min_percent": float(cases.observation_coverage_percent.min()),
               "coverage_max_percent": float(cases.observation_coverage_percent.max()),
               "new_fits": 0, "new_accounts": 0, "new_return_labels": 0,
               "new_strategy_CAGR": "NOT_COMPUTED", "new_strategy_Sharpe": "NOT_COMPUTED",
               "full_factor_score": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED",
               "next_action": "先补未绑定的不同证据及原公布版本/指数传导，建立条件评分预测完整用途与相同共同池对照，再登记一次完整账户比较。",
               "goal_achieved": False}
    write_json(OUT / "summary.json", summary)
    write_report(cases, catalog, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_report(cases, catalog, summary):
    lines = ["# 全因素联合讨论与评分：因素总表和30个原案例", "",
             "本轮按用户最新要求把所有主要因素纳入讨论，完成12组、" + str(len(catalog)) + "项目录及30个原案例的事实矩阵。综合分析可以开展；完整预测评分还没有被验证。", "",
             "0/20/40/60/80是固定解释工作等级，描述已确认的核心组合，不是胜率、目标收益或交易建议。估值、盈利、主线、真实流量等未完整绑定，因此不输出伪精确的全因素总分。", "",
             "原支持表和宏观表的部分观察采用16:00；本轮统一历史15:05，重新检查公告/调查修改/资金/融资时间。未认证历史首版；全部案例已被观察，只属开发解释。现金股息调整价与原始价格分别显示。", "",
             "## 全部主要因素及角色", "", "| 因素组 | 本组因素 | 作用与评分口径 |", "|---|---|---|"]
    for group, role, items in GROUPS:
        names = "、".join(name for name, _ in items)
        lines.append(f"| {group} | {names} | {role}；同组相近指标共同解释，不重复计独立票 |")
    lines += ["", "## 联合解释等级", "", "| 工作分 | 已确认组合 |", "|---:|---|",
              "| 0 | 日线MACD和五日动量都未显示修复 |", "| 20 | 仅部分日线修复；已有支持持仓遇到同工具明确兑现偏弱时也封顶20 |",
              "| 40 | 日线动量共同修复，但尚无本卡绑定的支持与价格区域接受 |",
              "| 60 | 日线修复＋原支持信息已公开＋价格越过原固定区域 |",
              "| 80 | 60分条件＋银行资金相对宽松、融资余额增长及量价参与共同确认 |",
              "| 100 | 尚未实现；还须不同的主线/盈利/估值/风险证据和完整可检验定义 |", "",
              "PMI低于50和周线仍弱描述政策修复背景，不能在所有阶段固定扣分。融资余额、ETF成交量、份额变化不是同一个变量。评分设计尚未证明对未来收益有排序能力。", "",
              "## 原12支持接受日与一次持仓复查", "",
              "| 日期 | 角色 | 原始收盘 | 工作分 | PMI订单 | 银行/融资状态 | 信息完整度 |", "|---|---|---:|---:|---:|---|---:|"]
    subset = cases[cases.case_role.ne("ORIGINAL_FIXED_KEY_DATE")]
    for row in subset.itertuples():
        score = "未知" if pd.isna(row.core_explanatory_score) else str(int(row.core_explanatory_score))
        pmi = "未知" if pd.isna(row.pmi_orders) else f"{row.pmi_orders:.1f}"
        lines.append(f"| {row.date:%Y-%m-%d} | {row.case_role} | {row.close:.3f} | {score} | {pmi} | {row.liquidity_state} | {row.observation_coverage_percent:.1f}% |")
    lines += ["", "## 原结果和反例", "",
              "原八技术＋六宏观联合树状态REJECTED_FIXED_MACRO_TECH_FIRST_PASSAGE_FULL_ACCOUNT_GATE_FAILED；原固定非线性评分状态COMPLETED_FIXED_NONLINEAR_SCORE_NO_RANKING_EDGE。本目录不改变这些用途的拒绝。", "",
              "2019年初与2023年6月均有弱订单、弱周线和日线修复；两者的银行资金、融资、相对成交量以及后续信用兑现状态不同。2023-06-20的调查在6月19日15:19才发布，不能放进6月15日的评分。此复查不将LPR调查当成6月13日OMO事前共识。", "",
              "工作等级只描述观测组合。2015年的部分组合较高也不能证明后续风险较低；30个日期均原已选案例，不用于计算新胜率或选交易阈值。所有原17关键日及12支持日均保存，没有仅展示盈利案例。", "",
              "## 尚需补齐及下一实验", "",
              "目录未绑定的因素保留UNKNOWN_NOT_ZERO。项目已有估值/盈利/行业等数据不等于这张卡已完成当时成员、公布钟及首版核对；晚修订的盈利不能加入早期预测。当前ETF份额只有旧日期基线，不能用来算历史进场当天的真实新流量。", "",
              "下一预测实验需先给出同一时点的完整输入合同、经济状态如何改变因素角色、唯一条件评分输出及成熟参考结果。比较纯价量、已失败宏观用途的保存对照和新不同信息用途；逐项去组对照检验新增信息，而不按结果挑指标/阶段/阈值。", "",
              "完整账户仍统一20万元、510300/CASH、日线与上一已完成周、次开盘、100份整数、T+1、原风险及两费用；收益/夏普同时提高、净pB>1和标准期望正、回撤约束及独立验证继续适用。此次0账户/拟合/新收益标签，完整目标未完成。", "",
              "信息完整度仅为已观察目录槽位比例，既不是预测置信度，也不是相互独立信息量。", "",
              "外部制度依据：[上交所ETF实物申赎说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734751.shtml)明确申赎可以改变份额；二级成交量不能直接替代新份额或人民币净流入。", "",
              "完整因素表、30案例表和逐因素矩阵位于results，来源与固定口径位于protocol.json。"]
    (OUT / "全因素联合讨论_评分与真实案例.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="全因素讨论与联合解释评分，不运行金融账户。")
    parser.add_argument("action", choices=("freeze", "run"))
    action = parser.parse_args().action
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
