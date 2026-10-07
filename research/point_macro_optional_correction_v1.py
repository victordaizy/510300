"""订单库存的可选残差修正：缺失时精确保留原模型，原全部成熟成员不删行。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.point_volatility_unit_exit_v1 import model_at_entry
from research.point_p02_exit_prediction_v1 import PERIODS, improvement_interval
from research.factor96_rapid_orders_inventory_v1 import parse_month


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_macro_optional_correction_v1"
OLD_FIELDS = ROOT / "reports/research/510300_factor96_rapid_orders_inventory_v1/source_fields_and_anchors.json"
VINTAGE = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet"
AUGUST_RESULT = ROOT / "reports/research/510300_manufacturing_price_transmission_source_v1/result.json"
FIELDS = ["ORDERS_INVENTORY_GAP", "ORDERS_INVENTORY_CHANGE3"]
KIND = "OPTIONAL_ORDERS_INVENTORY_RESIDUAL_CORRECTION"
STUDY = "510300_POINT_MACRO_OPTIONAL_CORRECTION_V1"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def source_paths() -> list[Path]:
    old = read(OLD_FIELDS)
    vintage = pd.read_parquet(VINTAGE)
    august = read(AUGUST_RESULT)["latest_source"]
    paths = [Path(__file__), ROOT / "tests/test_point_macro_optional_correction_v1.py",
             OLD_FIELDS, VINTAGE, AUGUST_RESULT,
             ROOT / "research/factor96_rapid_orders_inventory_v1.py",
             ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/within_cycle_exit_inputs_v1.py",
             ROOT / "research/point_volatility_unit_exit_v1.py",
             ROOT / "research/point_p02_exit_prediction_v1.py",
             ROOT / "research/point_account_cashflow_state_v1.py",
             ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    paths += [CURRENT / p for p in ["inputs/candidate_features.parquet", "inputs/within_models.json",
              "inputs/config/within_cycle_exit.json", "results/training_reference/samples.parquet",
              "results/training_reference/cycles.parquet"]]
    paths += [ROOT / row["raw_path"] for row in old]
    paths += [ROOT / str(p) for p in vintage.loc[~vintage.reference_period.isin([r["stat_month"] for r in old]), "raw_path"]]
    paths += [ROOT / august["raw_path"]]
    return sorted(set(paths))


def check_freeze() -> int:
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "可选宏观修正协议改变")
    for row in frozen["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "冻结来源变化：" + row["path"])
    return len(frozen["sources"])


def freeze() -> None:
    require(not (OUT / "freeze.json").exists(), "该唯一实验已冻结，不覆盖")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] >= 7, "必要时钟、缺失和固定截距测试未通过")
    require(tests["module_sha256"] == digest(Path(__file__)) and
            tests["test_sha256"] == digest(ROOT / "tests/test_point_macro_optional_correction_v1.py"), "测试代码变化")
    protocol = {
        "at": now(), "study": STUDY, "protocol_decision": "TECH.R116", "result_decision": "TECH.R117",
        "candidate_configurations": 1, "hypothesis": "已知订单库存水平和同口径三月变化能否增加原继续价值预测的信息。",
        "different_use": "旧月度高低筛选32账户零合格保持拒绝；此处只拟合固定原八项预测的周期内残差。",
        "scope": "日线/已完成周线；510300.SH及现金；只研究标的点位，长仓优先，无期权收益或实盘订单。",
        "field": "制造业当月新订单DI减产成品库存DI；三月变化要求四个连续月全部同方法组。",
        "source": "复用2015—2025的132份原提取记录；从本地原文新增解析2026年1—8月八份。八月新订单仅同页重复表一致，不冒充独立数值来源。",
        "clock": "原页面发布日期当日23:59:59，Asia/Shanghai；当前原点15:05；历史物理首发未认证，属于开发重建假设。",
        "latest_release_rule": "只取冻结已准入日志中原点之前最新报告；最新报告缺失或跨口径时不回退更旧报告。未来新增报告不得改变先前原点。",
        "source_boundaries": "全母集仍1507行。早期无报告、版本变更后三月差缺失等原值保持NULL；可选项可用不等于完整K04卡全部成员字段准入。DI不是订单数量、市场预期差或因果证明。",
        "version_assumption": "三月差不跨2019分类和2022样本变更；各口径DI水平统一缩放是固定模型假设，不证明版本完全可比。",
        "core": "原八项系数、截距、尺度及115可用/27不可用月全部原样保留，不重新拟合原模型。",
        "training": "原最近20个完整成熟周期、至少10周期100行，原全部训练行、目标和每周期总权重1不变；拟合日和入场首收盘锁定版本不变。",
        "normalizer": "只用当时训练中两字段均已知的行及其原权重计算均值和标准差，标准差为0则取1；z截到±5，再减已知行加权z均值。",
        "design": "已知分支phi=clip(z)-已知行加权clip(z)均值；未知分支phi=0定义为修正不活动，不把原未知字段填零。全训练权重均值phi=0。",
        "fit": "所有原行保留；在原周期内分别中心化phi和(y-原预测)，解beta=(Dphi'W Dphi+I2)^-1 Dphi'W D残差；alpha固定1，不增加全局截距。",
        "prediction": "两项当前原值均已知且训练修正已识别时原预测加phi·beta；否则直接返回原预测浮点值。原模型未知则双方未知。",
        "prediction_gate": "两个固定时期完整原配对、周期等权原收益MSE均严格改善，入场年块5000次seed51030099改进95%下界均>0。",
        "periods": PERIODS,
        "economic_gate": "预测门通过后才登记原20万元252日、BASE/STRESS及双时期账户；两时期净CAGR/夏普同时高于A，实际净pB>1、标准净期望>0、最大回撤<=10%，交易次数软目标。",
        "no_rescue": "不按结果改缺失分支、窗口、方法组规则、方向、alpha、clip、标签、训练成员、权重、样本起止或费用，不拼接子期。",
        "new_return_labels": 0, "network_requests": 0, "independent_validation": "NOT_ESTABLISHED",
        "history_role": "DEVELOPMENT_CALIBRATION", "global_DSR_PBO": "NOT_COMPUTED", "goal_achieved": False,
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    paths = source_paths() + [OUT / "tests_receipt.json"]
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]}, exclusive=True)
    print("订单库存唯一可选残差修正实验已冻结，尚未计算预测结果。")


def extract_august_orders(source: dict) -> float:
    raw = (ROOT / source["raw_path"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == source["raw_sha256"], "八月原文摘要不符")
    soup = BeautifulSoup(raw, "html.parser")
    target = "2026年8月"
    values = []
    compact = lambda text: re.sub(r"\s+", "", text)
    for table in soup.find_all("table"):
        rows = [[compact(c.get_text()) for c in tr.find_all(["td", "th"], recursive=False)] for tr in table.find_all("tr")]
        headers = [r for r in rows if "新订单" in r and all(k in r for k in ["生产", "原材料库存", "从业人员"])]
        current = [r for r in rows if r and r[0] == target]
        if headers:
            require(len(headers) == len(current) == 1, "八月制造业订单当月行不唯一")
            offset = len(current[0]) - len(headers[0])
            require(offset in (0, 1, 2), "八月表头宽度不一致")
            values.append(float(current[0][headers[0].index("新订单") + offset]))
    require(len(values) >= 1 and len(set(values)) == 1, "八月订单重复表不一致")
    return values[0]


def monthly_values(records: list[dict]) -> pd.DataFrame:
    monthly = pd.DataFrame([{k: v for k, v in r.items() if k != "anchors"} for r in records])
    monthly = monthly.sort_values("stat_month").reset_index(drop=True)
    require(not monthly.stat_month.duplicated().any(), "报告月份重复")
    monthly["published_at"] = pd.to_datetime(monthly.published_at, utc=True).dt.tz_convert("Asia/Shanghai")
    monthly["known_at"] = pd.to_datetime(monthly.known_at, utc=True).dt.tz_convert("Asia/Shanghai")
    require(monthly.known_at.is_monotonic_increasing and monthly.known_at.is_unique, "报告可用钟非唯一递增")
    expected = monthly.published_at.dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    require(monthly.known_at.equals(expected), "不允许提前使用当日公布数据")
    months = pd.PeriodIndex(monthly.stat_month, freq="M").asi8
    same = pd.Series(True, index=monthly.index)
    for step in (1, 2, 3):
        same &= monthly.method_group.eq(monthly.method_group.shift(step))
        same &= pd.Series(months, index=monthly.index).sub(pd.Series(months, index=monthly.index).shift(step)).eq(step)
    monthly[FIELDS[1]] = monthly[FIELDS[0]].diff(3).round(1).where(same)
    monthly["auxiliary_available"] = monthly[FIELDS].notna().all(axis=1)
    require(np.isfinite(monthly[FIELDS[0]]).all(), "已解析订单库存水平缺失")
    return monthly


def load_sources() -> tuple[list[dict], pd.DataFrame]:
    records = copy.deepcopy(read(OLD_FIELDS))
    require(len(records) == 132 and records[0]["stat_month"] == "2015-01" and records[-1]["stat_month"] == "2025-12", "旧132报告母集变化")
    old_months = {r["stat_month"] for r in records}
    vintage = pd.read_parquet(VINTAGE)
    fresh = vintage.loc[~vintage.reference_period.isin(old_months)].sort_values("reference_period")
    require(fresh.reference_period.tolist() == [f"2026-{m:02d}" for m in range(1, 8)], "继承2026七月度档案变化")
    for row in fresh.itertuples(index=False):
        require(digest(ROOT / row.raw_path) == row.source_hash, "新增月份原文摘要不符")
        parsed = parse_month(row)
        parsed["numeric_reference_status"] = "INHERITED_RELEASE_VINTAGE_NUMERIC_CHECK"
        records.append(parsed)
    august = read(AUGUST_RESULT)["latest_source"]
    require(august["stat_month"] == "2026-08", "继承新增源不是八月")
    row = SimpleNamespace(reference_period="2026-08", raw_path=august["raw_path"],
                          first_release_value=extract_august_orders(august), published_at=august["published_at"],
                          source_url=august["url"], source_hash=august["raw_sha256"])
    parsed = parse_month(row)
    require(parsed["method_group"] == august["method_group"], "八月方法组与原源不符")
    parsed["numeric_reference_status"] = "SAME_PAGE_SELF_CONSISTENCY_ONLY_NO_INDEPENDENT_NUMERIC_REFERENCE"
    records.append(parsed)
    require(len(records) == 140, "冻结报告总数不是140")
    return records, monthly_values(records)


def align_fields(states: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原状态身份重复")
    # Pandas可使用微秒或纳秒存储；和Timestamp.value比较前明确统一为纳秒。
    clocks = pd.DatetimeIndex(monthly.known_at).as_unit("ns").asi8
    result = []
    for row in states.itertuples(index=False):
        at = pd.Timestamp(row.origin).normalize().tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        index = int(np.searchsorted(clocks, at.value, side="right") - 1)
        source = monthly.iloc[index] if index >= 0 else None
        known = bool(source is not None and source.auxiliary_available)
        result.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
                       "origin_at": at, "stat_month": source.stat_month if source is not None else None,
                       "known_at": source.known_at if source is not None else pd.NaT,
                       "method_group": source.method_group if source is not None else None,
                       FIELDS[0]: float(source[FIELDS[0]]) if source is not None else np.nan,
                       FIELDS[1]: float(source[FIELDS[1]]) if known else np.nan,
                       "auxiliary_available": known,
                       "availability_status": "KNOWN_OPTIONAL_INPUT" if known else "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK"})
    return pd.DataFrame(result)


def design(rows: pd.DataFrame, model: dict) -> np.ndarray:
    raw = rows[FIELDS].to_numpy(float)
    available = np.isfinite(raw).all(axis=1)
    require(np.array_equal(available, rows.auxiliary_available.to_numpy(bool)), "原可选值与分支不符")
    phi = np.zeros((len(rows), 2), dtype=float)
    if model["identified"] and available.any():
        z = np.clip((raw[available] - np.asarray(model["mean"])) / np.asarray(model["scale"]), -5., 5.)
        phi[available] = z - np.asarray(model["clip_center"])
    return phi


def centered_system(rows: pd.DataFrame, core: dict, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    phi = design(rows, model)
    baseline = np.asarray([within_cycle_prediction(core, values) for values in rows[BASE_FEATURES].to_numpy(float)])
    residual = rows.target.to_numpy(float) - baseline
    weights = rows.sample_weight.to_numpy(float)
    ids = rows.cycle_id.to_numpy(int)
    dx, dy, groups = np.empty_like(phi), np.empty_like(residual), []
    for cycle_id in sorted(set(ids)):
        mask = ids == cycle_id
        require(np.allclose(weights[mask], 1. / mask.sum(), atol=1e-14, rtol=0), "原周期权重改变")
        xp = np.average(phi[mask], axis=0, weights=weights[mask])
        yr = float(np.average(residual[mask], weights=weights[mask]))
        dx[mask], dy[mask] = phi[mask] - xp, residual[mask] - yr
        groups.append({"cycle_id": int(cycle_id), "rows": int(mask.sum()), "design_mean": xp.tolist(), "residual_mean": yr})
    return dx, dy, weights, groups


def fit_optional(rows: pd.DataFrame, core: dict) -> dict:
    require(len(rows) > 0 and not rows.duplicated(["cycle_id", "origin_index"]).any(), "训练行为空或重复")
    require(np.isfinite(rows[BASE_FEATURES + ["target", "sample_weight"]].to_numpy(float)).all(), "原训练数值缺失")
    require((rows.sample_weight > 0).all(), "训练权重非法")
    raw = rows[FIELDS].to_numpy(float)
    available = np.isfinite(raw).all(axis=1)
    require(np.array_equal(available, rows.auxiliary_available.to_numpy(bool)), "训练缺失分支不符")
    mean, scale, center = np.zeros(2), np.ones(2), np.zeros(2)
    if available.any():
        w = rows.sample_weight.to_numpy(float)[available]
        mean = np.average(raw[available], axis=0, weights=w)
        scale = np.sqrt(np.average((raw[available] - mean) ** 2, axis=0, weights=w))
        scale = np.where(scale > 1e-12, scale, 1.)
        center = np.average(np.clip((raw[available] - mean) / scale, -5., 5.), axis=0, weights=w)
    model = {"kind": KIND, "features": FIELDS, "identified": bool(available.any()),
             "mean": mean.tolist(), "scale": scale.tolist(), "clip_center": center.tolist(),
             "coefficients": [0., 0.], "global_intercept": 0., "ridge_alpha": 1., "feature_clip": 5.,
             "training_rows": len(rows), "known_training_rows": int(available.sum()),
             "unknown_training_rows_retained": int((~available).sum())}
    dx, dy, weights, groups = centered_system(rows, core, model)
    gram = dx.T @ (weights[:, None] * dx) + np.eye(2)
    rhs = dx.T @ (weights * dy)
    beta = np.linalg.solve(gram, rhs)
    require(np.isfinite(beta).all(), "修正系数非有限值")
    require(np.max(np.abs(np.average(design(rows, model), axis=0, weights=weights))) < 1e-12, "修正加入全局截距")
    model.update(coefficients=beta.tolist(), cycle_centering=groups,
                 normal_equation_gradient_max=float(np.max(np.abs(gram @ beta - rhs))))
    return model


def predict_optional(core: dict, auxiliary: dict, base_values: list | np.ndarray, values: list | np.ndarray) -> tuple[float, float, str]:
    baseline = within_cycle_prediction(core, base_values)
    raw = np.asarray(values, float)
    require(raw.shape == (2,), "修正需要两项原值")
    require(auxiliary["kind"] == KIND and auxiliary["features"] == FIELDS, "修正模型身份不同")
    if not np.isfinite(raw).all() or not auxiliary["identified"]:
        return baseline, baseline, "EXACT_CORE_FALLBACK"
    phi = np.clip((raw - np.asarray(auxiliary["mean"])) / np.asarray(auxiliary["scale"]), -5., 5.) - np.asarray(auxiliary["clip_center"])
    correction = float(phi @ np.asarray(auxiliary["coefficients"]))
    return baseline, baseline + correction, "OPTIONAL_CORRECTION_AVAILABLE"


def training_identity(rows: pd.DataFrame, core: dict) -> str:
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight", "auxiliary_available"] + BASE_FEATURES + FIELDS
    raw = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    core_raw = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw + core_raw).hexdigest()


def period_results(paired: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    periods, tables = [], []
    for name, start, end in PERIODS:
        part = paired.loc[paired.origin.between(pd.Timestamp(start), pd.Timestamp(end))]
        original = part.baseline_prediction.notna()
        matched = part.loc[original & part.candidate_prediction.notna()].copy()
        matched["old_error"] = (matched.baseline_prediction - matched.target) ** 2
        matched["new_error"] = (matched.candidate_prediction - matched.target) ** 2
        group = matched.groupby("cycle_id").agg(baseline_mse=("old_error", "mean"), candidate_mse=("new_error", "mean"),
                     entry_year=("entry_year", "first"), rows=("origin_index", "size")).reset_index()
        group["improvement"] = group.baseline_mse - group.candidate_mse
        group["period"] = name
        interval = improvement_interval(group)
        baseline, candidate = float(group.baseline_mse.mean()), float(group.candidate_mse.mean())
        complete = len(matched) == int(original.sum()) and bool(original.any())
        passed = bool(complete and candidate < baseline and interval["low"] is not None and interval["low"] > 0)
        periods.append({"period": name, "original_available_rows": int(original.sum()), "paired_rows": len(matched),
                        "cycles": len(group), "complete_pair_coverage": complete,
                        "optional_input_known_rows": int(matched.auxiliary_available.sum()),
                        "exact_fallback_rows": int(matched.status.eq("EXACT_CORE_FALLBACK").sum()),
                        "baseline_raw_return_mse": baseline, "candidate_raw_return_mse": candidate,
                        "relative_mse_change": candidate / baseline - 1, "improvement_interval": interval,
                        "prediction_gate_passed": passed})
        tables.append(group)
    return periods, pd.concat(tables, ignore_index=True)


def original_training(states: pd.DataFrame, record: dict, cfg: dict) -> pd.DataFrame:
    rows, ids = training_rows(states, record["fit_index"], cfg)
    require(ids == record["training_cycles"] and len(rows) == record["training_rows"], "原全部成熟成员被改变")
    require((rows.exit_index <= record["fit_index"]).all() and
            (pd.to_datetime(rows.mature_date) <= pd.Timestamp(record["fit_origin"])).all(), "训练使用未成熟目标")
    return rows


def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一预测实验已开始，禁止科学重跑")
    source_count = check_freeze()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "source_count": source_count}, exclusive=True)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require(len(states) == 1507 and len(originals) == 142, "原快照母集数量变化")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]] == [20, 10, 100, 1., 5.], "原训练口径变化")
    sources, monthly = load_sources()
    field = align_fields(states, monthly)
    require(np.array_equal(states[["cycle_id", "origin_index"]], field[["cycle_id", "origin_index"]]), "来源与原状态顺序不同")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    write_json(OUT / "source_fields_and_anchors.json", sources, exclusive=True)
    save_table("原文月度字段与版本", monthly)
    save_table("原自然成员可选字段", field)
    write_json(OUT / "source_summary.json", {"reports": len(monthly), "old_extractions_reused": 132,
          "new_original_pages_parsed": 8, "first_month": "2015-01", "last_month": "2026-08",
          "original_natural_rows": 1507, "optional_input_known_rows": int(field.auxiliary_available.sum()),
          "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
          "complete_K04_all_member_field_admission": False, "functional_core_fallback": True,
          "historical_first_vintage_verified": False, "new_network_requests": 0}, exclusive=True)
    cache, candidates, receipts = {}, [], []
    for original in originals:
        record = copy.deepcopy(original)
        record["auxiliary_model"] = None
        if original["status"] == "FIT_COMPLETE":
            rows = original_training(states, original, cfg)
            key = training_identity(rows, original["model"])
            first = key not in cache
            if first:
                cache[key] = {"model": fit_optional(rows, original["model"]), "fit_origin": original["fit_origin"]}
            record.update(auxiliary_model=copy.deepcopy(cache[key]["model"]), input_identity=key,
                          reused=not first, first_auxiliary_estimation_origin=cache[key]["fit_origin"])
            receipts.append({"fit_origin": original["fit_origin"], "fit_index": original["fit_index"],
                     "training_rows": len(rows), "training_cycles": len(original["training_cycles"]),
                     "known_training_rows": record["auxiliary_model"]["known_training_rows"],
                     "unknown_training_rows_retained": record["auxiliary_model"]["unknown_training_rows_retained"],
                     "input_identity": key, "first_estimation": first,
                     "latest_training_exit_index": int(rows.exit_index.max())})
        candidates.append(record)
    write_json(OUT / "candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    paired = []
    for row in states.itertuples(index=False):
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old = model_at_entry(originals, entry)
        new = model_at_entry(candidates, entry)
        baseline, candidate, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        if old and old["status"] == "FIT_COMPLETE":
            require(old["latest_exit_index"] <= old["fit_index"] <= entry <= row.origin_index < row.early_exit_index < row.exit_index, "预测/成熟时钟错误")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "入场锁定原模型被改变")
            baseline, candidate, status = predict_optional(old["model"], new["auxiliary_model"],
                     [getattr(row, k) for k in BASE_FEATURES], [getattr(row, k) for k in FIELDS])
        paired.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
                      "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                      "target": float(row.target), "baseline_prediction": baseline, "candidate_prediction": candidate,
                      "auxiliary_available": bool(row.auxiliary_available), "status": status})
    paired = pd.DataFrame(paired)
    periods, losses = period_results(paired)
    passed = all(p["prediction_gate_passed"] for p in periods)
    known = paired.loc[paired.baseline_prediction.notna()]
    require(len(known) == 1010 and paired.baseline_prediction.isna().sum() == 497, "原预测可用母集变化")
    accounting = {"candidate_configurations": 1, "original_monthly_records": 142, "available_monthly_records": len(receipts),
                  "original_unknown_months_preserved": sum(r["model"] is None for r in candidates),
                  "distinct_auxiliary_training_inputs": len(cache), "auxiliary_coefficient_estimations": len(cache),
                  "core_model_reestimations": 0, "monthly_cache_reuses": len(receipts) - len(cache),
                  "new_return_labels": 0, "new_accounts": 0, "new_network_requests": 0}
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R117",
          "status": "PASS_FIXED_OPTIONAL_MACRO_PREDICTION_GATE" if passed else "REJECTED_FIXED_OPTIONAL_MACRO_PREDICTION_GATE_FAILED",
          "periods": periods, "prediction_gate_passed": passed, "accounting": accounting,
          "complete_original_policy_prediction_coverage": True, "complete_K04_all_member_field_admission": False,
          "all_natural_rows": len(paired), "paired_available_predictions": len(known),
          "original_unknown_predictions_preserved": int(paired.baseline_prediction.isna().sum()),
          "exact_fallback_available_predictions": int(known.status.eq("EXACT_CORE_FALLBACK").sum()),
          "prediction_sign_changes": int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum()),
          "economic_stage": "READY_FOR_SEPARATE_FIXED_ACCOUNT_REGISTRATION" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
          "net_sharpe": None, "net_cagr": None, "historical_first_vintage_verified": False,
          "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
          "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("原配对预测", paired), ("原周期等权预测误差", losses), ("全部成熟训练与复用", pd.DataFrame(receipts))]:
        save_table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def verify() -> None:
    source_count = check_freeze()
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    monthly = pd.read_parquet(OUT / "results/原文月度字段与版本.parquet")
    saved_fields = pd.read_parquet(OUT / "results/原自然成员可选字段.parquet")
    pd.testing.assert_frame_equal(align_fields(states, monthly), saved_fields)
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = saved_fields[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "原冻结模型或月度训练成员被修改")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "原无模型月份得到新模型")
            continue
        rows = original_training(states, old, cfg)
        key = training_identity(rows, old["model"])
        require(key == new["input_identity"], "训练数据指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()), "未知训练成员丢失")
            if known.any():
                raw = rows.loc[known, FIELDS].to_numpy(float)
                w = rows.loc[known, "sample_weight"].to_numpy(float)
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0) and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                        and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "训练尺度未还原")
            dx, dy, weights, groups = centered_system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = dx.T @ (weights * (dx @ beta-dy)) + beta
            err = float(np.max(np.abs(gradient)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=weights))))
            require(err < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "正规方程或固定截距不符")
            gradients.append(err)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/原配对预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "配对预测身份变化")
    max_error, fallback = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "原目标被改变")
        old = model_at_entry(originals, row.entry_index)
        new = model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "未知原预测被补值")
            continue
        a, b, status = predict_optional(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES], [getattr(state, k) for k in FIELDS])
        require(status == row.status, "预测分支变化")
        if status == "EXACT_CORE_FALLBACK":
            require(row.baseline_prediction == row.candidate_prediction == a == b, "未知分支不是精确原预测")
            fallback += 1
        max_error = max(max_error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
    periods, losses = period_results(paired)
    saved = read(OUT / "prediction_summary.json")
    require(periods == saved["periods"] and max_error == 0., "预测或误差区间未还原")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权预测误差.parquet"))
    verification = {"at": now(), "status": "PASS_SAVED_SOURCE_CLOCK_FIXED_CORE_ALL_MEMBER_NORMAL_EQUATION_PREDICTION_RECOMPUTATION",
                   "frozen_sources": source_count, "monthly_records": len(candidates), "distinct_normal_equations": len(verified),
                   "predictions_checked": len(paired), "exact_fallback_predictions": fallback,
                   "maximum_prediction_error": max_error, "maximum_normal_equation_gradient": max(gradients),
                   "maximum_global_design_mean": max(means), "cycle_losses_checked": len(losses),
                   "new_model_fits": 0, "new_accounts": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", verification, exclusive=True)
    print(json.dumps(verification, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="固定订单库存可选修正隔离实验")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
