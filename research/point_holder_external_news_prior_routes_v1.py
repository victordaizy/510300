"""合并七项持有人、休市外部信息和文本预期差的旧用途与来源核对。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_holder_external_news_prior_routes_v1"
PLAN = "reports/research/510300_point_next_information_intake_20261002/N01_N02_N05_O01_O03_O04_O06_prior_source_plan.json"
BASE = "reports/research"
FUND = f"{BASE}/510300_original_fund_subscription_reports_v1"
FLOW = f"{BASE}/510300_original_quarterly_flow_policy_v1"
GLOBAL = f"{BASE}/510300_overnight_global_information_v1"
CLOSURE = f"{BASE}/510300_fund_share_publication_receipts_closure_v1"
EPS = f"{BASE}/510300_eps_growth_disagreement_increment_v1"
NOVELTY = f"{BASE}/510300_eps_explicit_revision_novelty_diagnostic_v1"
PREFLIGHT = "reports/data_quality/510300_expectations_news_price_response_chain_pit_preflight_v1.json"
STATES = f"{BASE}/510300_point_current_observation_20261001/results/training_reference/samples.parquet"
MARKETS = {
    "GSPC": ("data/raw/cross_market_chart_ml_v1/GSPC_daily.parquet", "US"),
    "IXIC": ("data/raw/cross_market_chart_ml_v1/IXIC_daily.parquet", "US"),
    "HSI": ("data/raw/cross_market_chart_ml_v1/HSI_daily.parquet", "ASIA_DELAY"),
    "N225": ("data/raw/cross_market_chart_ml_v1/N225_daily.parquet", "ASIA_DELAY"),
    "VIX": ("data/raw/us_china_overnight_v1/VIX_daily.parquet", "US"),
}
IDS = ("N01", "N02", "N05", "O01", "O03", "O04", "O06")


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path: str) -> str:
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name: str, value: object) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pick(path: str, keys: tuple[str, ...]) -> dict:
    value = read(path)
    return {"source": path, **{k: value[k] for k in keys}}


def sources() -> list[str]:
    paths = [PLAN, *[r["path"] for r in read(PLAN)["sources"]],
             Path(__file__).relative_to(ROOT).as_posix(),
             "research/original_fund_quarterly_facts_v1.py", "research/us_china_overnight_binary_screen_v1.py",
             "config/510300_overnight_global_information_v1.json",
             "config/510300_original_quarterly_flow_policy_v1.json",
             "config/510300_expectations_news_price_response_chain_pit_preflight_v1.yaml",
             f"{FUND}/quarterly_facts_v1_1_result.json", f"{FUND}/quarterly_share_flow_facts_v1_1.parquet",
             f"{FLOW}/source_receipt.json", f"{FLOW}/result.json",
             f"{GLOBAL}/source_receipt.json", f"{GLOBAL}/result.json", f"{CLOSURE}/result.json",
             f"{EPS}/result.json", f"{NOVELTY}/result.json", PREFLIGHT, STATES,
             f"{BASE}/510300_mechanism_event_dossiers_v1/INDEX.md",
             *[p for p, _ in MARKETS.values()]]
    return list(dict.fromkeys(paths))


def cases() -> list[dict]:
    cards = {r["id"]: r for r in read(PLAN)["registered_cards"]}
    items = [
        ("N01", "同一历史主动权益基金池的净申赎与净值收益背离是否提供剩余价值信息。",
         "已修复56份510300原季报、55份事实与原目录时钟、55相邻季度恒等式；不是主动基金池，也未还原人民币现金流。旧价格加季度申赎主压力年化−1.624626%/夏普−0.020845/回撤48.295303%，原七候选16账户失败，不改季度规则营救。",
         "PASSIVE_SINGLE_ETF_QUARTERLY_SOURCE_NOT_ACTIVE_FUND_POOL_OLD_POLICY_FAILED"),
        ("N02", "真实公开净申购批次与当时NAV形成的存量成本代理是否解释回本赎回压力。",
         "原季度基金总申购/赎回不能拆成每日持有人批次；实际持有人与FIFO/比例赎回分配均未验证。另每日总份额响应首发时钟缺口已经封闭：2220已关联响应没有公开字段，不重新扫描或把检索日倒填。当前批次/NAV/分配合同未绑定，严格成本分布NOT_RUN。",
         "HOLDER_COHORT_NAV_AND_REDEMPTION_RULE_NOT_BOUND_PUBLICATION_CLOSURE_RETAINED"),
        ("N05", "中国休市期间、复市开盘前已结束的海外和汇率累计信息是否改变首反应后的价值。",
         "旧五海外时钟对齐模型主压力年化0.841700%/夏普0.130847/回撤52.016558%，原16候选没有达标。旧美时段中国ETF函数strict_preopen_interval_returns已经计算前一中国交易日至当前开盘前的跨多个美股交易日收益，跨休市累积并非新算法。五原源最新日期8月18—25日、当前原状态到9月24日；原完整海外/FX卡没有绑定，不能截短时期、填零或沿用t+1开盘前的信息到t收盘。",
         "OLD_GLOBAL_POLICY_FAILED_INTERVAL_ACCUMULATION_ALREADY_EXISTS_FULL_N05_NOT_ADMITTED"),
        ("O01", "同主体前90日固定词组相似度衡量的新事实是否提供增量信息。",
         "现有六机制病例来自已经拒绝的MACD×广度×下行波动预检，是看过收益后的病例，不是完整90日首见文本母集。七份选定研究报告的两叙述只复用旧数值，不能由数值新颖度0推断所有文本信息无效。完整去重、初版和固定文本算法合同未齐。",
         "SELECTED_CASES_NOT_NOVELTY_CORPUS_FULL_TEXT_CLOCK_NOT_BOUND"),
        ("O03", "同类型重复消息的新颖度和标准化首轮响应衰减是否改变继续价值。",
         "六事后病例或宏观少数节点不能构成同类型全部连续消息。旧预期—消息—响应PIT预检七门仅一通过，完整聚类、首版本与响应钟未过；本线固定类型/新颖度/首响应母集亦未绑定，不用事后涨跌重排事件。",
         "REPEAT_EVENT_UNIVERSE_AND_RESPONSE_CLOCK_NOT_BOUND"),
        ("O04", "固定可得来源、固定文本模型的观点方向分散度是否形成状态增量。",
         "旧两机构已报告年度EPS增长定义分歧有33成熟配对、误差增5.973152%，区间跨0且两完整年份仅一正，账户NOT_RUN；不是全部观点文本分散度。不同分析师/预测年/股本修正与新闻文本必须分开，不能改定义或权重复活旧失败，当前文本来源全集及首见时钟未绑定。",
         "OLD_NUMERIC_GROWTH_DISAGREEMENT_REJECTED_FULL_TEXT_DISPERSION_NOT_BOUND"),
        ("O06", "可核实传闻首发、正式确认和内容增量是否解释尚未兑现的影响。",
         "原卡明确仅合格前瞻样本；当前没有完整传闻首版及未获确认的失败传闻登记。正式公告档案和纯测量代码不能补出缺失的事前预期或传闻时钟，不以事后确认成功的六病例构造历史胜率。",
         "FORWARD_ONLY_RUMOR_CONFIRMATION_CONTRACT_UNESTABLISHED"),
    ]
    return [{"id": code, "name": cards[code]["name"], "hypothesis": hypothesis,
             "verification_method": "有限比对原卡、实际代码和保存终态；读取原表模式及海外日期，未计算新收益。",
             "result": result, "decision": decision, "field_admitted": False,
             "why": "不同原信息用途、完整来源和交易前版本、当前原全部成熟成员合同没有同时成立。拒绝具体旧表达，保留整体机制未知。",
             "revalidation": "需实质不同用途和完整合格原源，先登记时钟/成员及未知规则后固定检验；不删周期、换阈值/方向/窗口/费用营救。O06只接合格前瞻登记。"}
            for code, hypothesis, result, decision in items]


def account_quotes() -> list[dict]:
    rows = []
    for path in (f"{FLOW}/result.json", f"{GLOBAL}/result.json"):
        selected = [r for r in read(path)["primary"] if r["cost"] == "STRESS"]
        if len(selected) != 1:
            raise ValueError(f"原主压力账户非唯一：{path}")
        rows.append({"source": path, "annual_days": 242, "saved_row": selected[0],
                     "role": "原保存账户，非当前252日模型及本轮新账户"})
    return rows


def source_facts() -> dict:
    origins = pq.read_table(ROOT / STATES, columns=["origin"]).to_pandas().origin
    ranges = []
    for symbol, (path, clock) in MARKETS.items():
        f = pq.ParquetFile(ROOT / path)
        dates = pd.DatetimeIndex(pq.read_table(ROOT / path, columns=["date"]).to_pandas().date)
        if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
            raise ValueError(f"原海外日历不完整：{symbol}")
        if clock == "US":
            last = (dates[-1] + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("Asia/Shanghai")
        else:
            last = (dates[-1] + pd.Timedelta(days=1, hours=8)).tz_localize("Asia/Shanghai")
        ranges.append({"symbol": symbol, "path": path, "rows": f.metadata.num_rows,
                       "columns": f.schema_arrow.names, "first_date": str(dates[0].date()),
                       "last_date": str(dates[-1].date()), "inherited_clock": clock,
                       "last_saved_observation_available_clock_assumption": last.isoformat(),
                       "natural_states_on_later_dates": int((origins > last.tz_localize(None).normalize()).sum())})
    share = pq.ParquetFile(ROOT / f"{FUND}/quarterly_share_flow_facts_v1_1.parquet")
    return {
        "fund_source": pick(f"{FUND}/quarterly_facts_v1_1_result.json", ("status", "quarterly_reports",
            "reports_with_verified_share_flow_values", "reports_with_facts_and_official_clock", "unresolved_reports",
            "verified_adjacent_quarter_transitions", "cash_flow_amount_reconstructed", "prior_parser_outputs_preserved")),
        "fund_original_clock": "原目录/封面较晚日期、PDF元数据检查、后一个市场交易日开盘可用的保守假设，非历史HTTP首次回执证明。",
        "fund_schema": {"path": f"{FUND}/quarterly_share_flow_facts_v1_1.parquet",
                        "rows": share.metadata.num_rows, "columns": share.schema_arrow.names},
        "quarterly_policy": pick(f"{FLOW}/result.json", ("status", "number_of_candidates", "evaluation_accounts",
            "trained_model_count", "fit_opportunities", "evaluation_events", "fund_flow_scope", "independent_validation")),
        "global_source": read(f"{GLOBAL}/source_receipt.json"),
        "global_policy": pick(f"{GLOBAL}/result.json", ("status", "number_of_candidates", "trained_model_count",
            "historical_vendor_first_delivery_proven", "independent_validation")),
        "global_date_ranges": ranges,
        "original_natural_states": {"rows": len(origins), "first": str(origins.min()), "last": str(origins.max())},
        "date_range_role": "只读五原表日期与模式及原状态日期；晚于末观测的状态数是资料范围事实，不是正式字段支持或NYSE/汇率日历完整性证明。未读海外close价格。",
        "share_publication_closure": pick(f"{CLOSURE}/result.json", ("status", "historical_raw_response_count",
            "historical_publication_fields_found", "decision", "scope_limit")),
        "eps_disagreement": pick(f"{EPS}/result.json", ("status", "evaluation", "account_stage", "new_accounts_generated")),
        "numeric_novelty": pick(f"{NOVELTY}/result.json", ("status", "census_reports", "diagnostic_events",
            "new_original_numeric_forecast_values", "all_possible_report_text_features_rejected", "timing_alternative_tested")),
        "older_member_pit_chain": pick(PREFLIGHT, ("status", "metrics", "adjudication", "market_price_value_read",
            "return_prediction_allowed", "portfolio_evaluation_allowed")),
        "old_chain_scope_limit": "原七门是旧成员级预期—消息—响应链的裁决，不是所有公告及全部当前字段永久不准入证明。",
        "full_current_1507_115_field_support": "NOT_RUN_NO_COMPLETE_FIELD_BOUND",
        "closed_2220_responses_rescanned": False,
        "new_external_close_returns_computed": 0,
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError("本项已开始，禁止覆盖或重复运行")
    for row in read(PLAN)["sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"原提案来源变化：{row['path']}")
    manifest = [{"path": p, "sha256": sha(p), "bytes": (ROOT / p).stat().st_size} for p in sources()]
    outputs = {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
               "source_and_old_purpose_facts.json": source_facts()}
    if tuple(r["id"] for r in outputs["cases.json"]) != IDS:
        raise ValueError("七项固定范围改变")
    OUT.mkdir(parents=True)
    save("evidence_manifest.json", manifest)
    save("protocol.json", {"at": now(), "study_id": "510300_POINT_HOLDER_EXTERNAL_NEWS_PRIOR_ROUTES_V1",
         "technical_decision": "TECH.R115", "plan": PLAN, "scope": "七项有限旧用途、两原账户、六表模式及五日期范围。",
         "new_configs_fits_labels_predictions_accounts_collection": 0, "old_frozen_files_modified": False,
         "full_current_support_not_computed": True})
    for name, value in outputs.items():
        save(name, value)
    save("summary.json", {"at": now(), "study_id": "510300_POINT_HOLDER_EXTERNAL_NEWS_PRIOR_ROUTES_V1",
         "technical_decision": "TECH.R115", "status": "NOT_ADMITTED_SEVEN_HOLDER_EXTERNAL_NEWS_PRIOR_SOURCE_ROUTES",
         "completed_prior_routes": 7, "source_count": len(manifest), "saved_old_account_quotes": 2,
         "saved_schema_tables": 6, "saved_external_date_ranges": 5, "field_bound": False, "field_admitted": False,
         "current_full_member_support": "NOT_RUN_NO_COMPLETE_FIELD_BOUND", "new_strategy_configurations": 0,
         "new_model_fits": 0, "new_return_labels": 0, "new_predictions": 0, "new_accounts": 0,
         "network_requests": 0, "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
         "latest_actual_prediction_decision_unchanged": "TECH.R113", "independent_validation": "NOT_ESTABLISHED",
         "overall_overfit_removed": False, "goal_achieved": False,
         "goal_turn_classification": "progress", "blocked_audit_count": 0})
    paragraphs = "\n\n".join(f"**{r['id']} {r['name']}**\n\n{r['result']}" for r in outputs["cases.json"])
    report = ("# 七项持有人、休市外部信息及文本预期差来源核对\n\n"
              "七项完整卡均未准入，没有新增金融预测或收益夏普提升结果。上轮实际预测仍为"
              "TECH.R113入场ATR回撤的固定失败，不以来源核对替换其模型裁决。\n\n" + paragraphs +
              "\n\n五海外源日期与当前原状态的范围差只作来源诊断；不是完整原1507状态/115训练月的字段支持计算。"
              "不读取缺源后的海外价格，不补零、不截短样本，不扫描已关闭的2220份额响应。"
              "下一不同宏观六项按精确原卡核用途及来源：货币初值/共识/定义版本、社融与GDP当时值、"
              "订单库存初值、DR007真实起点、政策公告与利率响应母集。完整源资格成立才固定比较。\n")
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    print(f"七项有限来源核对完成：{len(manifest)}个证据对象，未准入；两原主压力账户摘录，新拟合/账户0。")


def verify() -> None:
    for row in read((OUT / "evidence_manifest.json").relative_to(ROOT).as_posix()):
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"冻结来源改变：{row['path']}")
    for name, expected in {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
                           "source_and_old_purpose_facts.json": source_facts()}.items():
        if read((OUT / name).relative_to(ROOT).as_posix()) != expected:
            raise ValueError(f"保存摘录不一致：{name}")
    save("saved_output_verification_receipt.json", {"at": now(),
         "status": "PASS_SAVED_SEVEN_HOLDER_EXTERNAL_NEWS_PRIOR_SOURCE_ROUTE_VERIFICATION",
         "cases": 7, "saved_original_account_rows": 2, "saved_schema_tables": 6,
         "saved_external_date_ranges": 5, "new_fits_labels_predictions_accounts_collection": 0,
         "limit": "只核原资料及摘录，不证明新字段资格、历史首次可得或策略有效。"})
    print("七分项、两原账户、六表模式和五日期范围核对通过；冻结策略保持。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="持有人、休市外部信息及文本预期差七项有限来源核对")
    parser.add_argument("operation", choices=("run", "verify"))
    args = parser.parse_args()
    run() if args.operation == "run" else verify()
