"""成分参与面六项有限路由与保存缓存覆盖上界；不构造新指标、预测或账户。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_breadth_prior_source_routes_v1"
INTAKE = "reports/research/510300_point_next_information_intake_20261002"
CURRENT = "reports/research/510300_point_current_observation_20261001"
PROGRAM = "reports/research/510300_factor96_program_v1"
MAJORITY = "reports/research/510300_breadth_majority_trend_v1"
RECLAIM = "reports/research/510300_factor96_internal_reclaim_v1_0_1"
SPEED = "reports/research/510300_factor96_rapid_breadth_speed_v1"
CROWD = "reports/research/510300_factor96_crowding_overlay_v1_0_1"
BREADTH = "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/internal_F_T_features.parquet"
WEIGHTED = "data/features/000300_official_weighted_breadth_daily.parquet"
WEIGHTS = "data/raw/constituents/000300_historical_weights.parquet"
PRICES = "data/raw/constituents/000300_constituent_daily.parquet"
IDS = [f"E0{i}" for i in range(1, 7)]


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def sha(path: str) -> str:
    with (ROOT / path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name: str, value) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_paths() -> list[str]:
    path = f"{INTAKE}/E01_E06_breadth_prior_source_routing_plan.json"
    plan = read(path)
    for source in plan["sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError(f"原提案来源变化：{source['path']}")
    paths = [path] + [entry["path"] for entry in plan["sources"]] + [
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        f"{PROGRAM}/strategy_progress.json", f"{MAJORITY}/result.json",
        "config/510300_breadth_majority_trend_v1.json",
        f"{RECLAIM}/protocol.json", f"{RECLAIM}/result.json", f"{RECLAIM}/source_receipt.json",
        f"{SPEED}/protocol.json", f"{SPEED}/result.json", f"{SPEED}/daily_signals.parquet",
        f"{CROWD}/protocol.json", f"{CROWD}/result.json", f"{CROWD}/source_receipt.json",
        f"{CROWD}/daily_features_lag1.parquet", BREADTH, WEIGHTED, WEIGHTS, PRICES,
        "reports/data_quality/000300_official_weighted_breadth_status.json",
        "reports/research/510300_macd_breadth_downside_preflight_v1_0_1.json",
        "reports/research/510300_up20_rare_event_forecast_v1_1_full_breadth_result.json",
        f"{CURRENT}/inputs/candidate_features.parquet",
        f"{CURRENT}/results/training_reference/samples.parquet", f"{CURRENT}/inputs/within_models.json",
    ]
    return list(dict.fromkeys(paths))


def old_quotes() -> list[dict]:
    path = f"{MAJORITY}/result.json"
    rows = [row for row in read(path)["all_metrics"] if row["cost"] == "STRESS"
            and row["model"] in ("BREADTH_MAJORITY_TREND", "PRICE20_ONLY")]
    if len(rows) != 2:
        raise ValueError("多数广度主方案和价格对照缺行")
    output = [{"source": path, "annual_days": 242, "saved_row": row} for row in rows]
    for path in (f"{RECLAIM}/result.json", f"{CROWD}/result.json"):
        rows = [row for row in read(path)["primary_rows"] if row["capital"] == 200000
                and row["cost"] == "STRESS" and row["policy"] == "FULL"]
        if len(rows) != 1:
            raise ValueError(f"旧FULL主行不唯一：{path}")
        output.append({"source": path, "annual_days": 242, "saved_row": rows[0]})
    path = f"{SPEED}/result.json"
    policies = ["MA20_CHANGE5_HIGH", "MA20_CHANGE5_LOW", "HIGH_LOW_CHANGE5_HIGH", "HIGH_LOW_CHANGE5_LOW",
                "E01_LEVEL_HIGH", "E01_CHANGE5_HIGH", "PRICE_CONTROL"]
    rows = [row for row in read(path)["primary_results"] if row["policy"] in policies
            and row["capital"] == 200000 and row["cost"] == "STRESS"]
    if len(rows) != len(policies):
        raise ValueError("参与速度四候选、两广度及一价格对照不完整")
    return output + [{"source": path, "annual_days": 242, "saved_row": row} for row in rows]


def source_masks() -> list[tuple[str, pd.DataFrame, pd.Series, str]]:
    breadth = pd.read_parquet(ROOT / BREADTH, columns=["date", "breadth20", "point_in_time_member_count",
        "return20_scoreable_member_count", "return20_coverage_ratio", "four_state_daily_coverage_state"])
    count = breadth.return20_scoreable_member_count
    mask = (np.isfinite(breadth.breadth20) & breadth.breadth20.between(0, 1)
            & breadth.point_in_time_member_count.eq(300) & count.between(294, 300) & count.mod(1).eq(0)
            & breadth.return20_coverage_ratio.ge(.98)
            & np.isclose(count / breadth.point_in_time_member_count, breadth.return20_coverage_ratio, rtol=0, atol=1e-12)
            & breadth.four_state_daily_coverage_state.eq("VIEW_ALLOWED"))
    speed = pd.read_parquet(ROOT / f"{SPEED}/daily_signals.parquet", columns=["date", "source_date", "source_known",
        "MA20_CHANGE5", "HIGH_LOW_CHANGE5", "E01_LEVEL", "E01_CHANGE5"])
    speed_mask = speed.source_known.eq(True) & np.isfinite(speed.iloc[:, 3:]).all(axis=1)
    if not (pd.to_datetime(speed.loc[speed_mask, "source_date"]) < pd.to_datetime(speed.loc[speed_mask, "date"])).all():
        raise ValueError("旧参与速度保存统计日不是严格前序")
    median = pd.read_parquet(ROOT / f"{CROWD}/daily_features_lag1.parquet", columns=["date", "stat_date",
        "internal_known", "common_members", "median20_current", "median20_prior5_same_members",
        "median20_change5", "ETF20_minus_median"])
    median_mask = median.internal_known.eq(True) & median.common_members.ge(294) & np.isfinite(median.iloc[:, 4:]).all(axis=1)
    if not (pd.to_datetime(median.loc[median_mask, "stat_date"]) < pd.to_datetime(median.loc[median_mask, "date"])).all():
        raise ValueError("旧中位成分保存统计日不是严格前序")
    weighted = pd.read_parquet(ROOT / WEIGHTED, columns=["date", "valid_for_direction_model"])
    return [
        ("E01_SAME_STAT_DAY_UPPER_BOUND", breadth, pd.Series(mask), "沿用旧独立breadth20自身98%和四态门；同统计日仅日期覆盖上界，不证明15:05可知。"),
        ("E03_SAVED_LAG1_INPUT_UPPER_BOUND", speed, speed_mask, "保存旧滞后一日四输入是否有效；不使用30/70分位、价格过滤或重新计算成分。"),
        ("E05_SAVED_LAG1_INPUT_UPPER_BOUND", median, median_mask, "保存旧滞后一日同成员中位数/变化/ETF差四字段有效；未适配为当前模型字段。"),
        ("OFFICIAL_WEIGHTED_SAME_STAT_DAY_UPPER_BOUND", weighted, weighted.valid_for_direction_model.eq(True),
         "旧95%权重覆盖标记的最乐观日期上界；不替换98%成员门，不把股票权重HHI当行业正贡献HHI。"),
    ]


def support_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    days = pd.read_parquet(ROOT / f"{CURRENT}/inputs/candidate_features.parquet", columns=["date"])
    samples = pd.read_parquet(ROOT / f"{CURRENT}/results/training_reference/samples.parquet",
                             columns=["cycle_id", "origin_index", "origin", "exit_index", "mature_date"])
    models = read(f"{CURRENT}/inputs/within_models.json")["models"]
    if (len(days), len(samples), len(models)) != (3488, 1507, 142):
        raise ValueError("当前冻结日历/成员/月数量变化")
    indexes = samples.origin_index.to_numpy(int)
    if not np.array_equal(pd.to_datetime(samples.origin).to_numpy(), pd.to_datetime(days.date.iloc[indexes]).to_numpy()):
        raise ValueError("原自然成员日期和索引不对应")
    daily = pd.DataFrame({"date": pd.to_datetime(days.date).dt.strftime("%Y-%m-%d")})
    natural = samples[["cycle_id", "origin_index"]].copy()
    natural["origin"] = pd.to_datetime(samples.origin).dt.strftime("%Y-%m-%d")
    monthly_rows, summaries = [], []
    for name, frame, mask, role in source_masks():
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError(f"旧保存源日期不是唯一递增：{name}")
        valid = pd.Series(mask.to_numpy(bool), index=dates).reindex(pd.DatetimeIndex(days.date)).eq(True).to_numpy()
        daily[name] = valid
        natural[name] = valid[indexes]
        rows = []
        for record in models:
            ids = record["training_cycles"]
            selected = samples[samples.cycle_id.isin(ids)]
            if len(ids) != len(set(ids)) or len(ids) != record["training_cycle_count"] or len(selected) != record["training_rows"]:
                raise ValueError("原训练成员数变化")
            if not (selected.exit_index.le(record["fit_index"]).all()
                    and pd.to_datetime(selected.mature_date).le(pd.Timestamp(record["fit_origin"])).all()):
                raise ValueError("原训练成熟上限不符")
            known = valid[selected.origin_index.to_numpy(int)]
            available = isinstance(record["model"], dict)
            rows.append({"source_bound": name, "fit_origin": record["fit_origin"], "fit_index": record["fit_index"],
                "original_model_available": available, "original_training_cycles": len(ids), "original_training_rows": len(selected),
                "date_upper_bound_supported_rows": int(known.sum()), "date_upper_bound_missing_rows": int((~known).sum()),
                "all_available_original_model_members_supported": bool(known.all()) if available else None,
                "first_publication_proved": False, "field_admitted": False})
        ready = [row for row in rows if row["original_model_available"]]
        if len(ready) != 115:
            raise ValueError("115原有模型/27原未知状态改变")
        summaries.append({"source_bound": name, "role": role, "saved_rows": len(frame), "saved_valid_rows": int(mask.sum()),
            "saved_start": str(dates.min().date()), "saved_end": str(dates.max().date()),
            "current_calendar_upper_bound_supported": int(valid.sum()), "current_natural_rows": 1507,
            "current_natural_upper_bound_supported": int(valid[indexes].sum()),
            "current_natural_upper_bound_missing": int((~valid[indexes]).sum()),
            "original_available_months": 115, "original_unavailable_months": 27,
            "fully_supported_original_available_months_upper_bound": sum(row["all_available_original_model_members_supported"] for row in ready),
            "minimum_missing_original_training_rows": min(row["date_upper_bound_missing_rows"] for row in ready),
            "physical_first_publication": "NOT_ESTABLISHED", "field_bound": False, "field_admitted": False})
        monthly_rows.extend(rows)
    return daily, natural, pd.DataFrame(monthly_rows), summaries


def source_facts() -> dict:
    raw = []
    for path, date_column in ((WEIGHTS, "trade_date"), (PRICES, "date")):
        file = pq.ParquetFile(ROOT / path)
        dates = pd.to_datetime(pd.read_parquet(ROOT / path, columns=[date_column])[date_column])
        raw.append({"path": path, "rows": file.metadata.num_rows, "date_start": str(dates.min().date()),
                    "date_end": str(dates.max().date()), "distinct_dates": dates.nunique(), "columns": file.schema_arrow.names,
                    "industry_history_or_first_publication_contract_established": False})
    macd = read("reports/research/510300_macd_breadth_downside_preflight_v1_0_1.json")
    rare = read("reports/research/510300_up20_rare_event_forecast_v1_1_full_breadth_result.json")
    strategy = {row["id"]: row for row in read(f"{PROGRAM}/strategy_progress.json") if row["id"] == "T04"}
    return {"raw_metadata": raw, "weighted_saved_quality": read("reports/data_quality/000300_official_weighted_breadth_status.json"),
            "weighted_scope": "权重36000行/120月2016-08-31至2026-07-31；价格484200行/1614日2019-12-23至2026-08-19。原权重按统计日<=当日选快照，retrieved_at不证明历史首次公布；两个所列schema没有行业历史。",
            "hhi_definition": "旧official_weight_hhi是股票权重平方和；E06是行业正收益贡献归一后平方和，不能相互替代。",
            "T04_saved_gate": strategy["T04"],
            "other_weighted_use": {"macd_status": macd["status"], "macd_passed": macd["passed"],
                "macd_adjudication": macd["adjudication"], "rare_status": rare["status"],
                "rare_primary_classification": rare["primary_result"]["classification"],
                "rare_probability_skill": rare["primary_result"]["probability_skill"],
                "rare_saved_stress": rare["primary_result"]["stress"]},
            "scope_limit": "核本轮有限指定源，不能据此断言全项目/外部没有行业、权重或公布证据。旧MACD/完整广度稀有上涨用途终态不重开，旧20k结果不与当前20万252排名。",
            "new_constituent_return_or_weight_calculation": 0, "new_industry_group_selection": 0,
            "new_future_labels_or_account_results": 0}


def cases() -> list[dict]:
    originals = {row["id"]: row for row in read(f"{INTAKE}/E01_E06_breadth_prior_source_routing_plan.json")["registered_cards"]}
    states = {row["id"]: row for row in read(f"{PROGRAM}/factor_progress.json") if row["id"] in IDS}
    items = [
        ("E01", "指数收益相近时，广度水平/变化能否补充参与面。", "旧多数上涨与MA20主方案压力CAGR−2.627534%、夏普−0.237576，好于去广度价格对照−5.290186%/−0.409612但目标失败；旧E01水平/变化HIGH及日更组件已有用途。原广度日期上界1051/1507、0/115，最少缺10原训练行。", "OLD_BREADTH_USES_TESTED_ALL_MEMBER_SOURCE_BOUND_FAILED"),
        ("E02", "指数二次试低而同成分破低更少，是否改善修复。", "旧T02同成员、同分母日内财富低价新低收缩已测，主期CAGR−0.690841%、夏普−0.616675、11周期；相对同源价格增量95%区间跨0。原18:00统计可知假设不能当本线15:05首版资格，零新增E02字段/支持检验。", "OLD_FIXED_T02_FAILED_DIFFERENT_CURRENT_INFORMATION_NOT_ADMITTED"),
        ("E03", "站上均线及新高减新低的参与恢复速度能否增加信息。", "旧四方向已初筛；MA20_CHANGE5_HIGH整体CAGR0.022214%、夏普0.021824且优于对照，但后期负，另三方向负，全部无继续资格。新高/新低是财富收盘而非日内极值；旧滞后四输入源上界755/1507、0/115，最少缺134原行。", "OLD_FOUR_CLOSE_BASED_DIRECTIONS_NOT_ACCEPTED_ALL_MEMBER_SOURCE_BOUND_FAILED"),
        ("E04", "前月固定领先行业能否向其他成员传播。", "原T04 NOT_RUN_PIT_WEIGHT_SOURCE_GATE保持；现有月度权重/价格元数据不提供历史行业及首次发布合同，不改等权完成原策略，不事后挑行业。", "T04_PIT_WEIGHT_AND_INDUSTRY_SOURCE_GATE_RETAINED"),
        ("E05", "指数与同成员中位数分歧是否改变持仓风险。", "旧T06同成员中位收益变化联合减仓已测，FULL主CAGR−1.188573%、夏普−0.636625，三个相对增量区间都跨0；ETF减中位差仅描述。旧滞后四字段源上界775/1507、0/115，最少缺122原行，不把组合失败当单字段无效。", "OLD_MEDIAN_COMPONENT_FAILED_ALL_MEMBER_SOURCE_BOUND_FAILED"),
        ("E06", "行业正贡献集中且总推动转弱是否增加风险信息。", "原行业正贡献HHI未验证；股票权重HHI/IF持仓HHI均非该定义。官方权重广度日期上界521/1507、0/115，最少缺284原行；原95%权重质量PASS不证明本用途历史行业/发布/全部成员资格。", "INDUSTRY_POSITIVE_CONTRIBUTION_NOT_BOUND_WEIGHTED_SOURCE_BOUND_FAILED"),
    ]
    return [{"id": factor, "name": originals[factor]["name"], "hypothesis": hypothesis,
        "verification_method": "合并读取原卡/真实旧协议源码和固定结果；11主账户原行，四份保存源可用标记的日期覆盖上界映射3488日/1507原状态/142原月，仅读成熟成员元数据，不读或重算目标。",
        "result": result, "routing_status": status, "original_card": originals[factor], "saved_program_status": states[factor],
        "acceptance_or_rejection": "接受旧用途、正负结果及日期上界事实；四源即使最乐观上界也各0/115，当前完整观测资格未建立，不准入字段/模型；不是全部广度或传播机制无效。",
        "revalidation": "只有实质不同用途且新的合格原源/行业/权重/首次发布及原全部成员完整支持才另立固定比较；不缩样本、填缺失、放宽294/300、换等权或重跑旧阈值/窗口/方向营救。",
        "field_bound": False, "field_admitted": False} for factor, hypothesis, result, status in items]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError("输出已存在，请verify，不覆盖原结果")
    paths = source_paths()
    manifest = [{"path": path, "sha256": sha(path)} for path in paths]
    daily, natural, monthly, summaries = support_tables()
    save("evidence_manifest.json", manifest)
    save("protocol.json", {"at": now(), "technical_decision": "TECH.R110", "cards": IDS,
        "classification": "FINITE_PRIOR_AND_SOURCE_UPPER_BOUND_DIAGNOSTIC_OLD_OUTCOMES_KNOWN_NOT_NEW_RETURN_EXPERIMENT",
        "source_comparison": "四保存缓存原可用标记映射原完整成员，E01/权重同统计日仅最乐观日期上界；E03/E05保留旧lag1，不计算新收益或改时滞。",
        "no_new_indicator_binding": True, "current_target_column_read": False, "new_return_models_or_accounts": 0,
        "model_maturity_metadata_only": True, "source_date_is_not_first_publication": True,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    save("cases.json", cases())
    save("saved_old_primary_account_quotes.json", old_quotes())
    save("source_facts.json", source_facts())
    save("source_upper_bound_summaries.json", summaries)
    for name, table in (("日线保存源覆盖上界.csv", daily), ("原自然成员源覆盖上界.csv", natural), ("原月度成员源覆盖上界.csv", monthly)):
        table.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    save("summary.json", {"at": now(), "study_id": "510300_POINT_BREADTH_PRIOR_SOURCE_ROUTES_V1", "technical_decision": "TECH.R110",
        "status": "NOT_ADMITTED_SIX_BREADTH_ROUTES_AND_FOUR_SAVED_SOURCE_UPPER_BOUNDS_FAILED",
        "source_count": len(paths), "completed_prior_routes": 6, "old_account_quotes": 11, "source_bounds": summaries,
        "coverage_table_rows": [len(daily), len(natural), len(monthly)],
        "original_available_models": 115, "original_unavailable_models_preserved": 27,
        "field_bound": False, "field_admitted": False, "new_strategy_configurations": 0,
        "new_ols_or_return_fits": 0, "new_return_labels": 0, "new_predictions": 0, "new_accounts": 0, "network_requests": 0,
        "current_field_support": "NOT_ADMITTED_UPPER_BOUND_ONLY", "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
        "goal_achieved": False, "goal_turn_classification": "progress", "blocked_audit_count": 0})
    text = "# 成分参与面六项旧用途与现存源覆盖上界（TECH.R110）\n\n六项当前不准入。四缓存在最乐观日期上界下，均无法完整支持任何一个原可用成熟训练月，四份均0/115；未删除原成员、缩样本或填未知，没有新增预测/账户及收益夏普结果。\n\n"
    text += "| 保存源上界 | 原缓存有效行 | 当前1507状态支持 | 115可用月完整支持 | 最少缺原训练行 |\n| --- | --- | --- | --- | --- |\n"
    for row in summaries:
        text += f"| {row['source_bound']} | {row['saved_valid_rows']}/{row['saved_rows']} | {row['current_natural_upper_bound_supported']}/1507 | {row['fully_supported_original_available_months_upper_bound']}/115 | {row['minimum_missing_original_training_rows']} |\n"
    text += "\n上界只回答原保存值/有效标记在哪些日期存在。E01/官方权重用同统计日最乐观映射，不能当15:05已知输入；E03/E05保持旧滞后一日，算法前序不证明实际历史首版。原142月仅读训练周期名单、行数和成熟上界，115有模型/27原无模型保留，没使用模型系数或样本目标。没有计算新成分收益、权重、行业选择或四张新因子。\n\n"
    for row in cases():
        text += f"- **{row['id']} {row['name']}**：{row['result']}\n\n"
    text += "接受旧多数广度虽减轻价格对照亏损仍目标失败、参与速度一个方向相对改善但后段为负的事实，不写成所有广度值都变差。新低收缩旧T02同源增量区间跨0；中位变化旧T06三个增量区间跨0，不能据多组件失败判单字段永远无效。原卡NOT_RUN与已做组件/初筛分别保留；不重开旧MACD×权重广度或完整广度稀有上涨的冻结失败。\n\n"
    text += "原权重36000行/120统计月2016-08-31至2026-07-31，所列价格484200行/1614日2019-12-23至2026-08-19；这两个schema未提供行业历史或实际首次发布时间。旧权重统计日<=当日及retrieved_at不等于发布合同。股票权重HHI、IF持仓HHI和行业正贡献HHI含义不同，不能换名。只在本轮指定源范围下未建立合同，不声称世界上没有其他数据。\n\n"
    text += "11原主账户行、四源三支持表和原月元数据保存核对一次；不为低风险事实摘录增加模拟测试。新字段/配置/OLS/回报拟合/标签/预测/账户/采集0，净收益夏普NOT_COMPUTED，独立验证未建立、过拟合未去除、完整目标未达。下一合并核IF日线五项及盈利现金信息六项的旧用途/来源；仍只研究510300点位，不交易辅助资产，原E03和原策略保持。\n\n"
    text += "直接证据：[六项路由](cases.json)、[四源上界数字](source_upper_bound_summaries.json)、[月度原成员表](原月度成员源覆盖上界.csv)、[旧主账户原行](saved_old_primary_account_quotes.json)、[来源及旧冻结用途](source_facts.json)、[保存核对](saved_output_verification_receipt.json)。\n"
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("六项参与面和四份缓存的完整成员上界已保存；四源均0/115，新模型/账户0。")


def verify() -> None:
    for entry in read((OUT / "evidence_manifest.json").relative_to(ROOT).as_posix()):
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"来源变化：{entry['path']}")
    daily, natural, monthly, summaries = support_tables()
    expected = {"cases.json": cases(), "saved_old_primary_account_quotes.json": old_quotes(),
                "source_facts.json": source_facts(), "source_upper_bound_summaries.json": summaries}
    for name, value in expected.items():
        if json.loads((OUT / name).read_text(encoding="utf-8")) != value:
            raise ValueError(f"保存摘录不一致：{name}")
    for name, table in (("日线保存源覆盖上界.csv", daily), ("原自然成员源覆盖上界.csv", natural), ("原月度成员源覆盖上界.csv", monthly)):
        if (OUT / name).read_text(encoding="utf-8-sig") != table.to_csv(index=False):
            raise ValueError(f"保存源上界表不一致：{name}")
    save("saved_output_verification_receipt.json", {"at": now(), "status": "PASS_SAVED_SIX_BREADTH_ROUTES_AND_FOUR_SOURCE_UPPER_BOUND_VERIFICATION",
        "old_account_rows_checked": 11, "table_rows_checked": [3488, 1507, 568],
        "monthly_metadata_checks": 568, "original_unknown_months_preserved": 27,
        "new_return_fits_labels_predictions_accounts": 0, "network_requests": 0,
        "limit": "只核上界/原元数据/保存事实，不证明首版、当前字段准入、因果或金融改善。"})
    print("六项/11旧行及3488、1507、568行保存表核对通过；未重算任何收益模型或账户。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="成分参与面有限旧用途和源覆盖上界核对")
    parser.add_argument("operation", choices=("run", "verify"))
    arguments = parser.parse_args()
    run() if arguments.operation == "run" else verify()
