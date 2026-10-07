"""原交易日资金政策利差的连续状态，对固定原预测做唯一可选残差比较。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.point_volatility_unit_exit_v1 import model_at_entry
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.factor96_funding_relief_v1 import policy_ledger
from research.point_optional_residual_model_v1 import design, fit, identity, predict, system


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_funding_optional_correction_v1"
DR_ROOT = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2"
POLICY_ROOT = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
DR = DR_ROOT / "dr007_daily_20150105_20260814.parquet"
POLICY = POLICY_ROOT / "pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet"
NODES = ROOT / "reports/research/510300_policy_information_clock_v1"
PROPOSAL = ROOT / "reports/research/510300_point_macro_optional_correction_v1/next_K05_source_and_model_proposal.json"
FIELDS = ["DR_POLICY_GAP_MEAN5", "DR_POLICY_GAP_MEAN5_CHANGE5"]
STUDY = "510300_POINT_FUNDING_OPTIONAL_CORRECTION_V1"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def policies() -> pd.DataFrame:
    raw = pd.read_parquet(POLICY)
    nodes = read(NODES / "inputs/policy_nodes.json")
    ledger = policy_ledger(raw, nodes)
    require(len(raw) == 25 and len(ledger) == 26, "原政策水平或实施节点数量变化")
    for row in ledger.itertuples(index=False):
        path = Path(row.raw_path)
        path = path if path.is_absolute() else ROOT / path
        require(digest(path) == row.raw_sha256, "政策实施原文摘要不符")
    return ledger


def sources() -> list[Path]:
    paths = [Path(__file__), ROOT / "research/point_optional_residual_model_v1.py",
             ROOT / "tests/test_point_funding_optional_correction_v1.py", PROPOSAL,
             DR, POLICY, DR_ROOT / "dr007_tushare_source_acquisition_manifest.json",
             POLICY_ROOT / "pboc_source_acquisition_manifest.json",
             POLICY_ROOT / "pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet",
             NODES / "inputs/policy_nodes.json",
             ROOT / "research/factor96_funding_relief_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
             ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
             ROOT / "research/point_p02_exit_prediction_v1.py",
             ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    paths += [CURRENT / p for p in ["inputs/candidate_features.parquet", "inputs/within_models.json",
               "inputs/config/within_cycle_exit.json", "results/training_reference/samples.parquet",
               "results/training_reference/cycles.parquet"]]
    for row in policies().itertuples(index=False):
        path = Path(row.raw_path)
        paths.append(path if path.is_absolute() else ROOT / path)
    return sorted(set(paths))


def check() -> int:
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "资金修正协议变化")
    for row in frozen["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "资金修正冻结源变化：" + row["path"])
    return len(frozen["sources"])


def freeze() -> None:
    require(not (OUT / "freeze.json").exists(), "唯一资金实验已冻结，不覆盖")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] >= 8, "资金字段必要测试未完成")
    require(tests["module_sha256"] == digest(Path(__file__)) and
            tests["model_sha256"] == digest(ROOT / "research/point_optional_residual_model_v1.py") and
            tests["test_sha256"] == digest(ROOT / "tests/test_point_funding_optional_correction_v1.py"), "测试实现变化")
    for row in read(PROPOSAL)["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "前述有限提案的原源发生变化")
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R118", "result_decision": "TECH.R119",
        "candidate_configurations": 1, "hypothesis": "资金价格相对已实施政策利率的连续压力与缓和能否增加原继续价值信息。",
        "different_use": "旧T14高低分位入场退出组合只有1闭合周期、点位失败保持；此处没有该规则的价格确认、分位阈值或账户营救。",
        "fields": FIELDS, "formula": "原点t使用原日历t-5..t-1五个DR日各自减该DR日收盘前已公布且已实施的政策七天利率的均值；变化为该均值减t-5原点对应均值，共需t-10..t-1十个原日历观测。单位百分点。",
        "clock": "DR源DR007.IB/DR007/weight，加权平均百分比；每个利率日期后的下一510300交易日09:30才可用，原状态15:05。政策原公布时刻及实施日均须成立。",
        "policy": "25个原操作变更及已核实R02实施事实；2024-09-27按23:59:59可用，不把9/24宣布目标提前当操作利率，也不把实施事实当独立政策意外。",
        "calendar": "完整2905条DR源均保存；2823原股票交易日完整，另82银行间日期保留原表。本定义使用原股票日历前十日的对应观测，银行间额外日期不改变该五日单位。",
        "missing": "应有日期、当日可知政策版本或十日窗口缺任一即可选输入未知，原值NULL、修正不活动，直接返回原预测；不插值、不拖末值、不回退较旧窗口。源截止8/14，之后仅合法下一开盘可用观测，不伪造9月资金值。",
        "core_and_training": "原1507状态、142月115可用27未知、八项系数/截距/尺度不动；全部原成熟训练行、目标、周期总权重1、最近20/最少10周期100行及实际入场锁定不变。",
        "model": "唯一两系数可选残差Ridge：已知行按原权重标准化、clip±5，再中心化令全训练设计均值0；未知设计为不活动分支0但原字段仍NULL。原全部行的设计和原预测残差分别周期内中心化，alpha1，新增全局截距0。",
        "prediction_gate": "两个固定时期原可用预测完整配对，周期等权原收益MSE严格改善，入场年份块5000次seed51030099改进95%下界均>0。",
        "periods": PERIODS, "complete_K05_all_member_field_admission": False,
        "economic_gate": "仅预测门通过后另登记原20万元252日BASE/STRESS两时期账户；净CAGR和净夏普均高于A，实际净pB>1、标准净期望>0、回撤<=10%，次数软目标。",
        "no_rescue": "不按结果改变五日、变化期、时钟、缺失分支、alpha/clip、方向、标签、成员、费用或子期；季末只记录已知日历，不临时加筛子。",
        "new_return_labels": 0, "network_requests": 0, "historical_first_vintage_verified": False,
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "history_role": "DEVELOPMENT_CALIBRATION", "orders_authorized": False, "goal_achieved": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    paths = sources() + [OUT / "tests_receipt.json"]
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]}, exclusive=True)
    print("资金价格连续状态的唯一可选残差实验已冻结，尚未计算预测结果。")


def funding_fields(market: pd.DataFrame, dr: pd.DataFrame, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(market.date)).as_unit("ns")
    require(dates.is_unique and dates.is_monotonic_increasing and not dates.hasnans, "原日历不唯一递增")
    require(market.symbol.eq("510300.SH").all(), "资金字段原标的不同")
    require(dr.date.is_unique and np.isfinite(dr.dr007.to_numpy(float)).all(), "DR日期重复或数值缺失")
    q = policy.copy().sort_values("known_at").reset_index(drop=True)
    q["known_at"] = pd.to_datetime(q.known_at, utc=True).dt.tz_convert("Asia/Shanghai")
    q["effective_date"] = pd.to_datetime(q.effective_date).dt.normalize()
    require(q.known_at.is_monotonic_increasing and np.isfinite(q.rate).all(), "政策水平时钟或数值非法")
    source = dr[["date", "dr007"]].copy().sort_values("date").reset_index(drop=True)
    source_dates = pd.DatetimeIndex(pd.to_datetime(source.date)).as_unit("ns")
    available_indexes = np.searchsorted(dates.asi8, source_dates.asi8, side="right")
    policy_known, policy_effective, rates = [], [], []
    for day in source_dates:
        end = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        eligible = q.loc[q.known_at.le(end) & q.effective_date.le(day)]
        if len(eligible):
            chosen = eligible.iloc[-1]
            policy_known.append(chosen.known_at)
            policy_effective.append(chosen.effective_date)
            rates.append(float(chosen.rate))
        else:
            policy_known.append(pd.NaT)
            policy_effective.append(pd.NaT)
            rates.append(np.nan)
    source["policy_known_at"] = pd.to_datetime(policy_known, utc=True).tz_convert("Asia/Shanghai")
    source["policy_effective_date"] = pd.to_datetime(policy_effective)
    source["policy_rate_percent"] = rates
    source["gap_pp"] = source.dr007 - source.policy_rate_percent
    available = [dates[int(i)].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
                 if i < len(dates) else pd.NaT for i in available_indexes]
    source["available_at"] = pd.to_datetime(available, utc=True).tz_convert("Asia/Shanghai")
    source["is_original_stock_session"] = source_dates.isin(dates)
    by_date = source.set_index("date")
    daily = pd.DataFrame({"date": dates, "origin_index": np.arange(len(dates))})
    daily["dr_stat_date"] = dates
    daily["dr007"] = by_date.dr007.reindex(dates).to_numpy(float)
    daily["policy_rate_percent"] = by_date.policy_rate_percent.reindex(dates).to_numpy(float)
    daily["gap_pp_on_stat_day"] = by_date.gap_pp.reindex(dates).to_numpy(float)
    daily["policy_known_at"] = by_date.policy_known_at.reindex(dates).array
    daily["quote_available_at"] = by_date.available_at.reindex(dates).array
    daily["latest_required_quote_date"] = daily.date.shift(1)
    daily["earliest_required_quote_date"] = daily.date.shift(10)
    daily[FIELDS[0]] = daily.gap_pp_on_stat_day.shift(1).rolling(5, min_periods=5).mean()
    daily[FIELDS[1]] = daily[FIELDS[0]] - daily[FIELDS[0]].shift(5)
    daily["auxiliary_available"] = np.isfinite(daily[FIELDS].to_numpy(float)).all(axis=1)
    daily["origin_at"] = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    for i in np.flatnonzero(daily.auxiliary_available):
        require(i >= 10, "资金窗口不足十个原日期")
        window = daily.iloc[i-10:i]
        require(window.quote_available_at.notna().all() and (window.quote_available_at <= daily.origin_at.iloc[i]).all(), "资金数据提前读取")
        require(window.policy_known_at.notna().all() and (window.policy_known_at <= daily.origin_at.iloc[i]).all(), "政策提前读取")
    return daily, source


def align(states: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    ids = states.origin_index.to_numpy(int)
    require((ids >= 0).all() and (ids < len(daily)).all(), "原状态越界")
    rows = daily.iloc[ids].reset_index(drop=True)
    require(np.array_equal(pd.DatetimeIndex(states.origin).as_unit("ns").asi8,
                           pd.DatetimeIndex(rows.date).as_unit("ns").asi8), "资金字段与原点日期不同")
    field = states[["cycle_id", "origin_index", "origin"]].reset_index(drop=True).copy()
    for name in FIELDS + ["auxiliary_available", "origin_at", "latest_required_quote_date", "earliest_required_quote_date"]:
        field[name] = rows[name].to_numpy()
    field["status"] = np.where(field.auxiliary_available, "KNOWN_OPTIONAL_FUNDING_INPUT", "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK")
    return field


def load_field() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "symbol"])
    dr = pd.read_parquet(DR)
    required = {"ts_code": "DR007.IB", "repo_maturity": "DR007", "provider_value_field": "weight",
                "value_semantics": "WEIGHTED_AVERAGE_RATE_PERCENT", "source_identity": "TUSHARE_PRO_REPO_DAILY_DR007_IB",
                "availability_rule": "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE", "missing_value_rule": "NO_VIEW_NO_INTERPOLATION"}
    for key, value in required.items():
        require(dr[key].eq(value).all(), "DR精确源合同不符：" + key)
    manifest = read(DR_ROOT / "dr007_tushare_source_acquisition_manifest.json")
    require(digest(DR) == manifest["curated_artifact"]["sha256"] and len(dr) == 2905, "DR原准入清单不符")
    acquisition = read(POLICY_ROOT / "pboc_source_acquisition_manifest.json")
    require(acquisition["observation_start"] == "2015-01-05" and acquisition["observation_cutoff"] == "2026-08-14", "政策完整来源界限变化")
    policy = policies()
    daily, source = funding_fields(market, dr, policy)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=["cycle_id", "origin_index", "origin"])
    return align(states, daily), daily, source, policy


def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "固定资金预测已开始，不重跑")
    source_count = check()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": source_count}, exclusive=True)
    field, daily, source, policy = load_field()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require(len(states) == 1507 and len(originals) == 142, "资金实验原母集变化")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]] == [20, 10, 100, 1., 5.], "资金实验原训练规则变化")
    require(np.array_equal(states[["cycle_id", "origin_index"]], field[["cycle_id", "origin_index"]]), "资金字段身份不同")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    for name, frame in [("全部DR原观察与可用钟", source), ("政策实际水平与实施钟", policy),
                         ("原日历资金状态", daily), ("原自然成员可选资金字段", field)]:
        table(name, frame)
    write_json(OUT / "source_summary.json", {"status": "PASS_EXACT_LOCAL_SOURCE_AND_OPTIONAL_FUNCTION_CLOCK",
         "source_DR_rows": len(source), "original_stock_session_quotes": int(source.is_original_stock_session.sum()),
         "extra_interbank_dates_retained": int((~source.is_original_stock_session).sum()),
         "policy_records": len(policy), "all_natural_rows": len(field),
         "optional_input_known_rows": int(field.auxiliary_available.sum()), "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
         "first_DR_date": "2015-01-05", "last_DR_date": "2026-08-14", "complete_K05_all_member_field_admission": False,
         "physical_first_vintage_verified": False, "new_network_requests": 0}, exclusive=True)
    cache, candidates, receipts = {}, [], []
    for original in originals:
        record = copy.deepcopy(original)
        record["auxiliary_model"] = None
        if original["status"] == "FIT_COMPLETE":
            rows = original_training(states, original, cfg)
            key = identity(rows, original["model"], FIELDS)
            first = key not in cache
            if first:
                cache[key] = {"model": fit(rows, original["model"], FIELDS), "fit_origin": original["fit_origin"]}
            record.update(auxiliary_model=copy.deepcopy(cache[key]["model"]), input_identity=key, reused=not first,
                          first_auxiliary_estimation_origin=cache[key]["fit_origin"])
            receipts.append({"fit_origin": original["fit_origin"], "fit_index": original["fit_index"],
                  "training_rows": len(rows), "training_cycles": len(original["training_cycles"]),
                  "known_training_rows": record["auxiliary_model"]["known_training_rows"],
                  "unknown_training_rows_retained": record["auxiliary_model"]["unknown_training_rows_retained"],
                  "latest_training_exit_index": int(rows.exit_index.max()), "input_identity": key, "first_estimation": first})
        candidates.append(record)
    write_json(OUT / "candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    predictions = []
    for row in states.itertuples(index=False):
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old, new = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        a, b, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        if old and old["status"] == "FIT_COMPLETE":
            require(old["latest_exit_index"] <= old["fit_index"] <= entry <= row.origin_index < row.early_exit_index < row.exit_index, "资金预测或目标时钟错误")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "资金实验改变入场锁定原版本")
            a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(row, k) for k in BASE_FEATURES], [getattr(row, k) for k in FIELDS], FIELDS)
        predictions.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
                 "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                 "target": float(row.target), "baseline_prediction": a, "candidate_prediction": b,
                 "auxiliary_available": bool(row.auxiliary_available), "status": status})
    paired = pd.DataFrame(predictions)
    periods, losses = period_results(paired)
    known = paired.loc[paired.baseline_prediction.notna()]
    require(len(known) == 1010 and int(paired.baseline_prediction.isna().sum()) == 497, "资金实验原预测母集变化")
    passed = all(p["prediction_gate_passed"] for p in periods)
    accounting = {"candidate_configurations": 1, "monthly_records": 142, "mature_monthly_records": len(receipts),
           "original_unknown_months_preserved": sum(r["model"] is None for r in candidates),
           "distinct_auxiliary_training_inputs": len(cache), "auxiliary_coefficient_estimations": len(cache),
           "monthly_cache_reuses": len(receipts) - len(cache), "core_model_reestimations": 0,
           "new_return_labels": 0, "new_accounts": 0, "new_network_requests": 0}
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R119",
         "status": "PASS_FIXED_FUNDING_OPTIONAL_PREDICTION_GATE" if passed else "REJECTED_FIXED_FUNDING_OPTIONAL_PREDICTION_GATE_FAILED",
         "periods": periods, "prediction_gate_passed": passed, "accounting": accounting,
         "all_natural_rows": len(paired), "paired_available_predictions": len(known), "original_unknown_predictions_preserved": 497,
         "exact_fallback_available_predictions": int(known.status.eq("EXACT_CORE_FALLBACK").sum()),
         "prediction_sign_changes": int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum()),
         "complete_K05_all_member_field_admission": False, "complete_original_policy_prediction_coverage": True,
         "economic_stage": "READY_FOR_SEPARATE_FIXED_ACCOUNT_REGISTRATION" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
         "net_cagr": None, "net_sharpe": None, "historical_first_vintage_verified": False,
         "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False,
         "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("原配对资金修正预测", paired), ("原周期等权预测误差", losses), ("全部成熟训练与复用", pd.DataFrame(receipts))]:
        table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def verify() -> None:
    source_count = check()
    field, daily, source, policy = load_field()
    for name, frame in [("全部DR原观察与可用钟", source), ("政策实际水平与实施钟", policy),
                         ("原日历资金状态", daily), ("原自然成员可选资金字段", field)]:
        pd.testing.assert_frame_equal(frame, pd.read_parquet(OUT / "results" / (name + ".parquet")))
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, global_means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "原资金对照模型或成员被改变")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "原冷启动月份得到新模型")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"], FIELDS)
        require(key == new["input_identity"], "资金训练身份变化")
        if key not in verified:
            model = new["auxiliary_model"]
            require(model["features"] == FIELDS and model["training_rows"] == len(rows), "资金模型字段或全成员不同")
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["unknown_training_rows_retained"] == int((~known).sum()), "资金未知原训练行丢失")
            if known.any():
                raw, w = rows.loc[known, FIELDS].to_numpy(float), rows.loc[known, "sample_weight"].to_numpy(float)
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw - mu) ** 2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw - mu) / sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0) and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                        and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "资金尺度还原失败")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T @ (w * (dx @ beta - dy)) + beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "资金正规方程或截距不符")
            gradients.append(gradient)
            global_means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/原配对资金修正预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "资金配对预测身份改变")
    max_error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "资金原目标变化")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "资金未知原模型被补齐")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES], [getattr(state, k) for k in FIELDS], FIELDS)
        require(status == row.status, "资金预测分支变化")
        max_error = max(max_error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "资金缺失不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and max_error == 0., "资金预测误差或区间未还原")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权预测误差.parquet"))
    result = {"at": now(), "status": "PASS_SAVED_FUNDING_SOURCE_CLOCK_FIXED_CORE_ALL_MEMBER_MODEL_AND_LOSS_RECOMPUTATION",
              "frozen_sources": source_count, "DR_rows_recomputed": len(source), "daily_fields_recomputed": len(daily),
              "original_state_fields_recomputed": len(field), "policy_records_recomputed": len(policy),
              "monthly_records": len(candidates), "distinct_normal_equations": len(verified), "paired_predictions_checked": len(paired),
              "exact_fallback_predictions": fallbacks, "maximum_prediction_error": max_error,
              "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(global_means),
              "cycle_losses_checked": len(losses), "new_model_fits": 0, "new_accounts": 0,
              "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="固定资金利差连续状态的可选残差实验")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
