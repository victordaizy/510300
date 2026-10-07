"""H03四象限及同合约基差变化：固定原模型之外的唯一可选残差实验。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.point_volatility_unit_exit_v1 import model_at_entry
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_optional_residual_model_v1 import design, system


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_h03_optional_correction_v1"
OLD = ROOT / "reports/research/510300_if_open_interest_increment_v1"
CFG_PATH = ROOT / "config/510300_if_open_interest_increment_v1.json"
PROPOSAL = ROOT / "reports/research/510300_point_funding_optional_correction_v1/next_H03_prior_and_source_proposal.json"
FUTURES_ROOT = ROOT / "data/raw/futures/cffex_if_contract_history_v1_0_1"
RAW_FIELDS = ["OI_LOG_CHANGE5", "SPOT_LOG_RETURN5", "SAME_CONTRACT_BASIS_LOG_CHANGE5"]
QUADRANTS = ["OI_UP_PRICE_UP", "OI_UP_PRICE_DOWN", "OI_DOWN_PRICE_UP", "OI_DOWN_PRICE_DOWN", "OBSERVED_FLAT"]
FIELDS = ["Q_" + q for q in QUADRANTS] + [RAW_FIELDS[2]]
KIND = "FIXED_CORE_OPTIONAL_H03_QUADRANTS_AND_BASIS_RIDGE"
STUDY = "510300_POINT_H03_OPTIONAL_CORRECTION_V1"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def encode(oi: float, price: float) -> tuple[str, list[float]]:
    if not np.isfinite([oi, price]).all():
        return "UNKNOWN", [np.nan] * 5
    if oi == 0. or price == 0.:
        name = "OBSERVED_FLAT"
    else:
        name = "OI_" + ("UP" if oi > 0. else "DOWN") + "_PRICE_" + ("UP" if price > 0. else "DOWN")
    return name, [float(name == q) for q in QUADRANTS]


def calculate(market: pd.DataFrame, daily: pd.DataFrame, spot: pd.DataFrame,
              comparable_start: str = "2016-01-04") -> pd.DataFrame:
    """仅原日历的完整真实合约行；基差配对不用未来主力、最后出现日或到期元数据。"""
    dates = pd.DatetimeIndex(pd.to_datetime(market.date)).as_unit("ns")
    require(dates.is_unique and dates.is_monotonic_increasing and market.symbol.eq("510300.SH").all(), "H03原日历或标的不同")
    require(not daily.duplicated(["date", "symbol"]).any() and daily.symbol.str.fullmatch(r"IF\d{4}").all(), "H03真实合约身份重复或错误")
    require(spot.date.is_unique and spot.symbol.eq("000300.SH").all(), "H03现货身份或日期错误")
    raw = daily.copy()
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    spot_close = spot.set_index("date").close.reindex(dates)
    groups = {pd.Timestamp(day): part.sort_values("symbol") for day, part in raw.groupby("date")}
    totals, complete, positive_price, sets = [], [], [], []
    for day in dates:
        rows = groups.get(day)
        if rows is None:
            totals.append(np.nan)
            complete.append(False)
            positive_price.append(False)
            sets.append(tuple())
            continue
        oi = rows.open_interest.to_numpy(float)
        full = len(rows) == 4 and np.isfinite(oi).all() and (oi >= 0).all()
        totals.append(float(oi.sum()) if full else np.nan)
        complete.append(bool(full))
        prices = rows.close.to_numpy(float)
        positive_price.append(bool(full and np.isfinite(prices).all() and (prices > 0).all()))
        sets.append(tuple(rows.symbol))
    records = []
    start = pd.Timestamp(comparable_start)
    for i, day in enumerate(dates):
        oi_change, price_return, basis = np.nan, np.nan, np.nan
        eligible, window = [], dates[max(0, i-5):i+1]
        if i >= 5:
            before = dates[i-5]
            if np.isfinite([totals[i], totals[i-5]]).all() and min(totals[i], totals[i-5]) > 0:
                oi_change = float(np.log(totals[i]/totals[i-5]))
            if np.isfinite([spot_close.iloc[i], spot_close.iloc[i-5]]).all() and min(spot_close.iloc[i], spot_close.iloc[i-5]) > 0:
                price_return = float(np.log(spot_close.iloc[i]/spot_close.iloc[i-5]))
            if before >= start and all(positive_price[i-5:i+1]) and np.isfinite(price_return):
                # 当前日正持仓和过去六日实际出现即可；到期零持仓行仍进入总量与完整四行检查。
                current = groups[day]
                eligible = [r.symbol for r in current.itertuples(index=False) if r.open_interest > 0
                            and all(r.symbol in sets[j] for j in range(i-5, i+1))]
                if eligible:
                    first = groups[before].set_index("symbol").close.loc[eligible].to_numpy(float)
                    last = current.set_index("symbol").close.loc[eligible].to_numpy(float)
                    basis = float(np.log(last/first).mean()-price_return)
        quadrant, flags = encode(oi_change, price_return)
        source_available = dates[i+1].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5) if i+1 < len(dates) else pd.NaT
        record = {"source_date": day, "source_index": i, "available_at_assumed": source_available,
                  "reported_contract_rows": len(groups[day]) if day in groups else 0,
                  "complete_four_contracts": complete[i], "aggregate_OI": totals[i],
                  "contract_symbols": "|".join(sets[i]), "observed_zero_OI_rows": int(groups[day].open_interest.eq(0).sum()) if day in groups else 0,
                  "contract_set_changed": bool(i and sets[i] and sets[i-1] and sets[i] != sets[i-1]),
                  "window_start_date": dates[i-5] if i >= 5 else pd.NaT,
                  "same_price_contract_count": len(eligible), "same_price_contract_symbols": "|".join(eligible),
                  RAW_FIELDS[0]: oi_change, RAW_FIELDS[1]: price_return, RAW_FIELDS[2]: basis, "quadrant": quadrant}
        record.update(dict(zip(FIELDS[:5], flags)))
        record["auxiliary_available"] = bool(np.isfinite([record[k] for k in FIELDS]).all())
        records.append(record)
    return pd.DataFrame(records)


def align(states: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in states.itertuples(index=False):
        origin = int(row.origin_index)
        require(0 <= origin < len(source) and pd.Timestamp(row.origin) == source.source_date.iloc[origin], "H03状态日期或索引不同")
        previous = source.iloc[origin-1] if origin > 0 else None
        at = pd.Timestamp(row.origin).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        if previous is not None:
            require(previous.available_at_assumed <= at, "H03假设时钟提前")
        record = {"cycle_id": int(row.cycle_id), "origin_index": origin, "origin": row.origin, "origin_at": at,
                  "source_date": previous.source_date if previous is not None else pd.NaT,
                  "available_at_assumed": previous.available_at_assumed if previous is not None else pd.NaT,
                  "quadrant": previous.quadrant if previous is not None else "UNKNOWN"}
        for key in RAW_FIELDS + FIELDS[:5]:
            record[key] = float(previous[key]) if previous is not None else np.nan
        known = bool(np.isfinite([record[k] for k in FIELDS]).all())
        record.update(auxiliary_available=known, status="KNOWN_OPTIONAL_H03_INPUT" if known else "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK")
        records.append(record)
    return pd.DataFrame(records)


def fit(rows: pd.DataFrame, core: dict) -> dict:
    require(len(rows) > 0 and not rows.duplicated(["cycle_id", "origin_index"]).any(), "H03原训练身份重复或为空")
    require(np.isfinite(rows[BASE_FEATURES + ["target", "sample_weight"]].to_numpy(float)).all() and (rows.sample_weight > 0).all(), "H03原训练缺失或权重非法")
    raw = rows[FIELDS].to_numpy(float)
    known = np.isfinite(raw).all(axis=1)
    require(np.array_equal(known, rows.auxiliary_available.to_numpy(bool)), "H03训练分支不一致")
    mean, scale, center = np.zeros(6), np.ones(6), np.zeros(6)
    if known.any():
        weights = rows.sample_weight.to_numpy(float)[known]
        mean = np.average(raw[known], axis=0, weights=weights)
        sd = np.sqrt(np.average((raw[known]-mean)**2, axis=0, weights=weights))
        scale = np.where(sd > 1e-12, sd, 1.)
        center = np.average(np.clip((raw[known]-mean)/scale, -5., 5.), axis=0, weights=weights)
    model = {"kind": KIND, "features": FIELDS, "identified": bool(known.any()), "mean": mean.tolist(),
             "scale": scale.tolist(), "clip_center": center.tolist(), "global_intercept": 0., "ridge_alpha": 1., "feature_clip": 5.,
             "training_rows": len(rows), "known_training_rows": int(known.sum()), "unknown_training_rows_retained": int((~known).sum())}
    dx, dy, weights = system(rows, core, model)
    gram, rhs = dx.T @ (weights[:, None]*dx) + np.eye(6), dx.T @ (weights*dy)
    beta = np.linalg.solve(gram, rhs)
    require(np.isfinite(beta).all(), "H03辅助系数非法")
    mean_error = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=weights))))
    require(mean_error < 1e-12, "H03修正加入全局截距")
    model.update(coefficients=beta.tolist(), global_design_mean_max=mean_error,
                 normal_equation_gradient_max=float(np.max(np.abs(gram@beta-rhs))))
    return model


def predict(core: dict, model: dict, base: list | np.ndarray, raw: list | np.ndarray) -> tuple[float, float, str]:
    require(model["kind"] == KIND and model["features"] == FIELDS, "H03模型身份不同")
    baseline = within_cycle_prediction(core, base)
    x = np.asarray(raw, float)
    require(x.shape == (6,), "H03模型需要固定六项设计")
    if not np.isfinite(x).all() or not model["identified"]:
        return baseline, baseline, "EXACT_CORE_FALLBACK"
    phi = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]), -5., 5.)-np.asarray(model["clip_center"])
    return baseline, baseline + float(phi@np.asarray(model["coefficients"])), "OPTIONAL_CORRECTION_AVAILABLE"


def identity(rows: pd.DataFrame, core: dict) -> str:
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight", "auxiliary_available"] + BASE_FEATURES + FIELDS
    raw = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    metadata = {"kind": KIND, "fields": FIELDS, "core": core, "alpha": 1., "clip": 5.}
    return hashlib.sha256(raw+json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def paths() -> list[Path]:
    config = read(CFG_PATH)
    receipt = read(ROOT / config["inputs"]["if_receipt"])
    result = [Path(__file__), ROOT / "tests/test_point_h03_optional_correction_v1.py", CFG_PATH, PROPOSAL,
              ROOT / "research/point_optional_residual_model_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
              ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
              ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
              ROOT / "research/point_p02_exit_prediction_v1.py", ROOT / "research/if_open_interest_increment_v1.py",
              ROOT / "scripts/download_cffex_if_true_term_structure_inputs_v1_0_1.py",
              ROOT / "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json",
              ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json",
              OLD / "result.json", OLD / "SOURCE_NOTES.md", OLD / "if_source_features.parquet",
              OUT / "prior_source_and_definition_review.json", OUT / "tests_receipt.json"]
    result += [ROOT / config["inputs"][k] for k in ["if_daily", "if_expiry", "if_receipt", "spot"]]
    result += [CURRENT / name for name in ["inputs/candidate_features.parquet", "inputs/within_models.json",
               "inputs/config/within_cycle_exit.json", "results/training_reference/samples.parquet", "results/training_reference/cycles.parquet"]]
    result += [FUTURES_ROOT / "monthly_archives" / (month+".zip") for month in receipt["source"]["archive_sha256"]]
    return sorted(set(result))


def check() -> int:
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "H03冻结协议改变")
    for row in frozen["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "H03冻结来源变化："+row["path"])
    return len(frozen["sources"])


def freeze() -> None:
    require(not (OUT / "freeze.json").exists(), "唯一H03协议已冻结，不覆盖")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] >= 8 and tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_h03_optional_correction_v1.py"), "H03必要测试或版本不符")
    prior = read(OUT / "prior_source_and_definition_review.json")
    require(prior["different_target_and_conditional_model_bound"] and not prior["old_failure_reopened"], "H03不同用途未成立")
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R120", "result_decision": "TECH.R121",
       "candidate_configurations": 1, "hypothesis": "同日OI和现货五日方向的四象限及同合约基差变化，是否增加当前原继续价值的条件信息。",
       "raw_fields": RAW_FIELDS, "model_features": FIELDS,
       "encoding": "源日log(sumOI_s/sumOI_s-5)与log(spot_s/spot_s-5)按严格正负固定四类；任何一个确知为0是第五FLAT类；未知不是FLAT。五类one-hot及同合约基差变化共六项设计。原三个连续字段分别保存。",
       "basis": "当前OI>0且过去原六日均实际出现的全部同一IF合约等权平均log(F_s/F_s-5)，减同期现货log收益；不读取未来主力、最后出现日或到期表来选择。这是名义基差变化，不是扣分红/融资成本套利。",
       "source": "复用15860真实日合约记录/3965日期/199代码，每源日四行，零OI到期行保留总量。现货000300.SH缓存为观察；价格比较仍从2016-01-04起，六原日历窗缺源则未知。",
       "clock": "原源日s的数据假设到下一原交易日t的15:05可用，t15:05形成状态后下一合法开盘才可应用；同原旧一日延迟研究假设，非逐日物理发布时间或首版证明。只用t-1的应有源行，不拖8/12末值补后续日。",
       "different_use": "原普通岭M0/M1五日独立往返、OI单日加方向交互失败保持；原IF_REL5已含本基差变化，不称新信息源。本次唯一固定四象限类别对原自然继续目标做周期内残差，不因仅五日窗口不同准入。",
       "core_and_members": "原八项/截距/尺度、1507状态、142月115可用27未知、全部成熟成员/目标/周期总权重1、最近20/最少10周期100行、入场锁定全保留。",
       "fit": "六项已知设计按原训练权重标准化、clip±5并中心化；未知修正分支0但原未知字段仍NULL。全部原训练行的设计及原预测残差周期内中心化，alpha1、全局新截距0，六辅助系数一次估计。原模型不重新拟合。",
       "missing": "合约母集不完整/价格不可比/原六日缺源/现货缺失等保持原值未知；完整输入不成立直接返回原预测，原模型未知双方未知。完整函数不等于完整H03卡字段或经济认证。",
       "prediction_gate": "两原固定时期完整配对、周期等权原收益MSE严格改善，入场年块5000次seed51030099改进95%下界均>0。",
       "periods": PERIODS, "economic_gate": "预测门过后才另冻结原20万元252日BASE/STRESS双期账户；净CAGR/净夏普均高于A、实际净pB>1、标准净期望>0、回撤<=10%、次数软目标。",
       "no_rescue": "不变五日/六源日、类别/平手规则、方向、alpha/clip、合约集合、缺失、目标、成员、成本或样本起止；不挑基差/到期或类目子样本营救，不事后拼接R117/R119。",
       "market_network_requests": 0, "new_return_labels": 0, "physical_first_vintage_verified": False,
       "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
       "global_DSR_PBO": "NOT_COMPUTED", "orders_authorized": False, "futures_trading_authorized": False, "goal_achieved": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths()]}, exclusive=True)
    print("H03固定四象限及基差条件残差协议已冻结，尚未读取本次预测结果。")


def load_fields() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = read(CFG_PATH)
    receipt = read(ROOT / config["inputs"]["if_receipt"])
    require(receipt["status"] == "SUCCESS_CFFEX_IF_CONTRACT_HISTORY_ACQUIRED", "H03原官方回执未通过")
    for name in ["if_daily", "if_expiry"]:
        require(digest(ROOT / config["inputs"][name]) == receipt["outputs"][config["inputs"][name]], "H03原输出哈希不符")
    for month, value in receipt["source"]["archive_sha256"].items():
        require(digest(FUTURES_ROOT / "monthly_archives" / (month+".zip")) == value, "H03原月包摘要不符")
    market = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "symbol"])
    daily = pd.read_parquet(ROOT / config["inputs"]["if_daily"])
    spot = pd.read_parquet(ROOT / config["inputs"]["spot"])
    require(len(daily) == 15860 and daily.date.nunique() == 3965 and daily.symbol.nunique() == 199
            and daily.groupby("date").size().eq(4).all(), "H03原合约母集数量不符")
    source = calculate(market, daily, spot, config["if_comparable_price_start"])
    # 旧基差确为已用信息，必须保存这个等价核对；不调用旧预测/标签/拟合。
    old = pd.read_parquet(OLD / "if_source_features.parquet", columns=["date", "IF_REL5"])
    matched = source.merge(old, left_on="source_date", right_on="date", how="left", validate="one_to_one")
    valid = matched[RAW_FIELDS[2]].notna() & matched.IF_REL5.notna()
    error = float(np.max(np.abs(matched.loc[valid, RAW_FIELDS[2]]-matched.loc[valid, "IF_REL5"])))
    require(error <= 1e-12 and valid.sum() > 2000, "H03原基差等价性未成立")
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=["cycle_id", "origin_index", "origin"])
    field = align(states, source)
    summary = {"status": "PASS_REAL_CONTRACT_SOURCE_AND_FIXED_OPTIONAL_FUNCTION_DEVELOPMENT_CLOCK",
          "raw_IF_rows": len(daily), "raw_IF_days": daily.date.nunique(), "raw_IF_contracts": daily.symbol.nunique(),
          "cached_spot_rows": len(spot), "last_IF_date": str(daily.date.max().date()), "last_spot_date": str(spot.date.max().date()),
          "price_comparable_start": config["if_comparable_price_start"], "original_state_rows": len(field),
          "optional_input_known_rows": int(field.auxiliary_available.sum()), "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
          "observed_quadrant_counts": field.loc[field.auxiliary_available, "quadrant"].value_counts().to_dict(),
          "same_basis_as_old_IF_REL5_rows": int(valid.sum()), "maximum_basis_equivalence_error": error,
          "complete_H03_all_member_field_admission": False, "source_clock_is_assumption": True,
          "physical_first_vintage_verified": False, "new_market_requests": 0}
    return field, source, summary


def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一H03实验已开始，不重复")
    source_count = check()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": source_count}, exclusive=True)
    field, source, source_summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require(len(states) == 1507 and len(originals) == 142, "H03原母集改变")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]] == [20, 10, 100, 1., 5.], "H03原训练口径改变")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    table("原日历真实合约条件字段", source)
    table("原自然成员H03可选字段", field)
    write_json(OUT / "source_summary.json", source_summary, exclusive=True)
    cache, candidates, receipts = {}, [], []
    for old in originals:
        record = copy.deepcopy(old)
        record["auxiliary_model"] = None
        if old["status"] == "FIT_COMPLETE":
            rows = original_training(states, old, cfg)
            key = identity(rows, old["model"])
            first = key not in cache
            if first:
                cache[key] = {"model": fit(rows, old["model"]), "fit_origin": old["fit_origin"]}
            record.update(auxiliary_model=copy.deepcopy(cache[key]["model"]), input_identity=key, reused=not first,
                          first_auxiliary_estimation_origin=cache[key]["fit_origin"])
            receipts.append({"fit_origin": old["fit_origin"], "fit_index": old["fit_index"], "training_rows": len(rows),
                             "training_cycles": len(old["training_cycles"]), "known_training_rows": record["auxiliary_model"]["known_training_rows"],
                             "unknown_training_rows_retained": record["auxiliary_model"]["unknown_training_rows_retained"],
                             "input_identity": key, "first_estimation": first})
        candidates.append(record)
    write_json(OUT / "candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    predictions = []
    for row in states.itertuples(index=False):
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old, new = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        a, b, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        if old and old["status"] == "FIT_COMPLETE":
            require(old["latest_exit_index"] <= old["fit_index"] <= entry <= row.origin_index < row.early_exit_index < row.exit_index, "H03预测成熟钟改变")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "H03改变原入场锁定模型")
            a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(row, k) for k in BASE_FEATURES], [getattr(row, k) for k in FIELDS])
        predictions.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
                     "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                     "target": float(row.target), "baseline_prediction": a, "candidate_prediction": b,
                     "auxiliary_available": bool(row.auxiliary_available), "status": status})
    paired = pd.DataFrame(predictions)
    periods, losses = period_results(paired)
    known = paired.loc[paired.baseline_prediction.notna()]
    require(len(known) == 1010 and paired.baseline_prediction.isna().sum() == 497, "H03原预测可用成员改变")
    passed = all(row["prediction_gate_passed"] for row in periods)
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R121",
       "status": "PASS_FIXED_H03_QUADRANT_PREDICTION_GATE" if passed else "REJECTED_FIXED_H03_QUADRANT_PREDICTION_GATE_FAILED",
       "periods": periods, "prediction_gate_passed": passed,
       "accounting": {"candidate_configurations": 1, "auxiliary_design_columns": 6, "original_monthly_records": 142,
               "available_monthly_records": len(receipts), "original_unknown_months_preserved": sum(r["model"] is None for r in candidates),
               "distinct_auxiliary_training_inputs": len(cache), "auxiliary_coefficient_estimations": len(cache),
               "monthly_cache_reuses": len(receipts)-len(cache), "core_model_reestimations": 0,
               "new_return_labels": 0, "new_accounts": 0, "new_market_requests": 0},
       "all_natural_rows": len(paired), "paired_available_predictions": len(known), "original_unknown_predictions_preserved": 497,
       "exact_fallback_available_predictions": int(known.status.eq("EXACT_CORE_FALLBACK").sum()),
       "prediction_sign_changes": int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum()),
       "economic_stage": "READY_FOR_SEPARATE_FIXED_ACCOUNT_REGISTRATION" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
       "net_sharpe": None, "net_cagr": None, "complete_H03_all_member_field_admission": False,
       "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
       "physical_first_vintage_verified": False, "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("原配对H03条件预测", paired), ("原周期等权预测误差", losses), ("全部成熟训练与复用", pd.DataFrame(receipts))]:
        table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def verify() -> None:
    source_count = check()
    field, source, source_summary = load_fields()
    pd.testing.assert_frame_equal(field, pd.read_parquet(OUT / "results/原自然成员H03可选字段.parquet"))
    pd.testing.assert_frame_equal(source, pd.read_parquet(OUT / "results/原日历真实合约条件字段.parquet"))
    require(source_summary == read(OUT / "source_summary.json"), "H03源与等价性摘要未还原")
    originals, candidates = read(CURRENT / "inputs/within_models.json")["models"], read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "H03原模型或月度成员被修改")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "H03原无模型月份被补")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"])
        require(key == new["input_identity"], "H03训练指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()), "H03未知原训练行丢失")
            if known.any():
                raw, w = rows.loc[known, FIELDS].to_numpy(float), rows.loc[known, "sample_weight"].to_numpy(float)
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0) and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                        and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "H03尺度不能还原")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T@(w*(dx@beta-dy))+beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "H03正规方程或全局截距改变")
            gradients.append(gradient)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/原配对H03条件预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "H03配对原身份改变")
    max_error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "H03改变原标签")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "H03原未知预测被填")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES], [getattr(state, k) for k in FIELDS])
        require(status == row.status, "H03预测分支改变")
        max_error = max(max_error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "H03未知不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and max_error == 0., "H03配对预测或区间未还原")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权预测误差.parquet"))
    result = {"at": now(), "status": "PASS_SAVED_H03_REAL_CONTRACT_CLOCK_FIXED_CORE_ALL_MEMBER_NORMAL_EQUATION_AND_PREDICTION_RECOMPUTATION",
              "frozen_sources": source_count, "source_calendar_rows_recomputed": len(source), "original_fields_recomputed": len(field),
              "monthly_records": len(candidates), "distinct_normal_equations": len(verified), "predictions_checked": len(paired),
              "exact_fallback_predictions": fallbacks, "maximum_prediction_error": max_error,
              "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(means),
              "cycle_losses_checked": len(losses), "new_model_fits": 0, "new_accounts": 0,
              "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="H03固定四象限及基差条件残差实验")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
