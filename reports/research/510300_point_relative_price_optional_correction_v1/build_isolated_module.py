"""生成完整隔离模块，复用原固定金融比较流程，不修改父实验。"""
from pathlib import Path
import ast

ROOT = Path.cwd()
DEST = ROOT / "research/point_relative_price_optional_correction_v1.py"
PREFIX = '''"""510300相对510500原价格状态的固定两系数持有退出比较。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.point_optional_residual_model_v1 import (
    design, system, fit as optional_fit, predict as optional_predict, identity as optional_identity, KIND)
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_volatility_unit_exit_v1 import model_at_entry

CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_relative_price_optional_correction_v1"
OTHER = ROOT / "reports/research/510300_csi300_csi500_relative_price_increment_v1"
COMPARISON = ROOT / "data/raw/all_etf_momentum_v1r/fund_daily_checkpoints/510500_SH.parquet"
MASTER = ROOT / "data/raw/all_etf_momentum_v1/fund_master.parquet"
FIELDS = ["RELATIVE_RAW_PRICE_RETURN_5", "RELATIVE_RAW_PRICE_RETURN_20"]
STUDY = "510300_POINT_RELATIVE_PRICE_OPTIONAL_CORRECTION_V1"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def table(name, frame):
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\\n")

def validate_raw(raw):
    require(raw.ndim == 2 and raw.shape[1] == 2, "相对价格固定两列。")
    require((np.isfinite(raw) | np.isnan(raw)).all(), "无穷值不是合法缺失，不填补。")
    return np.isfinite(raw).all(axis=1)

def validate_dates(frame):
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    require(len(dates) > 0 and not dates.hasnans and dates.is_unique
            and dates.is_monotonic_increasing and dates.tz is None, "原日期必须非空、递增、唯一且无时区。")
    require(dates.equals(dates.normalize()), "原交易日不得夹带盘中时间。")
    return dates

def build_field(stock, comparison):
    dates, other_dates = validate_dates(stock), validate_dates(comparison)
    require(stock.symbol.eq("510300.SH").all(), "执行标的必须为510300.SH。")
    require(comparison.ts_code.eq("510500.SH").all(), "观察来源必须为510500.SH，不能替代为000905.SZ。")
    primary = stock.close.to_numpy(float)
    require(np.isfinite(primary).all() and (primary > 0).all(), "原510300收盘价非法。")
    other_raw = comparison.ts_close.to_numpy(float)
    require((np.isfinite(other_raw) | np.isnan(other_raw)).all()
            and (other_raw[np.isfinite(other_raw)] > 0).all(), "原510500报价非法，缺价保持NaN。")
    secondary = pd.Series(other_raw, index=other_dates).reindex(dates).to_numpy(float)
    daily = pd.DataFrame({"date": dates, "original_510300_close": primary,
                          "original_510500_close": secondary})
    for field, window in zip(FIELDS, [5, 20], strict=True):
        p = pd.Series(primary)
        q = pd.Series(secondary)
        complete = q.notna().rolling(window + 1, min_periods=window + 1).sum().eq(window + 1)
        value = (p / p.shift(window) - 1.) - (q / q.shift(window) - 1.)
        daily[field] = value.where(complete).to_numpy(float)
    daily["known_at"] = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    daily["latest_source_index"] = np.arange(len(daily))
    daily["earliest_5_source_index"] = np.arange(len(daily)) - 5
    daily["earliest_20_source_index"] = np.arange(len(daily)) - 20
    daily["auxiliary_available"] = validate_raw(daily[FIELDS].to_numpy(float))
    return daily

def align(states, daily):
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原自然身份重复。")
    indices = states.origin_index.to_numpy(int)
    require((indices >= 0).all() and (indices < len(daily)).all(), "原自然索引越界。")
    field = daily.iloc[indices].reset_index(drop=True).copy()
    require(np.array_equal(pd.to_datetime(states.origin).to_numpy(dtype="datetime64[ns]"),
                           pd.to_datetime(field.date).to_numpy(dtype="datetime64[ns]")), "原日期与索引不一致。")
    field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
    field.insert(1, "origin_index", indices)
    field["origin_at"] = pd.DatetimeIndex(field.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    require((field.known_at <= field.origin_at).all(), "价格公开钟晚于原决定钟。")
    field["optional_status"] = np.where(field.auxiliary_available, "KNOWN_OPTIONAL_RELATIVE_RAW_PRICE",
                                        "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK")
    return field

def load_fields():
    stock = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "close", "symbol"])
    comparison = pd.read_parquet(COMPARISON, columns=["date", "ts_close", "ts_code"])
    master = pd.read_parquet(MASTER)
    identities = master.loc[master.ts_code.isin(["510300.SH", "510500.SH"])]
    require(len(identities) == 2 and identities.ts_code.is_unique, "两个ETF主表身份缺失或重复。")
    daily = build_field(stock, comparison)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    require(len(stock) == 3488 and len(states) == 1507, "原日线或自然母集变化。")
    field = align(states, daily)
    other = pd.read_parquet(OTHER / "两ETF原价与固定过去相对状态.parquet")
    overlap = daily.set_index("date").loc[pd.DatetimeIndex(other.date)]
    require(np.array_equal(overlap.original_510300_close.to_numpy(float), other.original_510300_close.to_numpy(float)),
            "另一用途与原技术线510300原价不一致，不拼接来源。")
    np.testing.assert_allclose(overlap.original_510500_close.to_numpy(float), other.original_510500_close.to_numpy(float),
                               rtol=0, atol=0, equal_nan=True)
    old_fields = ["510300相对510500原价格五日变化差", "510300相对510500原价格二十日变化差"]
    np.testing.assert_allclose(overlap[FIELDS].to_numpy(float), other[old_fields].to_numpy(float),
                               rtol=0, atol=1e-13, equal_nan=True)
    summary = {"status": "PASS_ORIGINAL_ETF_RELATIVE_RAW_PRICE_OPTIONAL_FUNCTION_ONLY",
               "daily_rows": len(daily), "comparison_source_rows": len(comparison),
               "comparison_source_start": str(comparison.date.min().date()),
               "comparison_source_end": str(comparison.date.max().date()),
               "same_fixed_E70_fields_overlap_rows": len(other),
               "known_daily_rows": int(daily.auxiliary_available.sum()), "original_state_rows": len(field),
               "optional_known_rows": int(field.auxiliary_available.sum()),
               "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
               "partial_known_original_state_rows": int((field[FIELDS].notna().any(axis=1) & ~field.auxiliary_available).sum()),
               "physical_first_vintage_verified": False, "other_branch_E71_failure_reversed": False,
               "complete_relative_price_all_member_field_admission": bool(field.auxiliary_available.all()),
               "source_clock": "RECONSTRUCTED_SESSION_CLOSE_1500_ORIGINAL_DECISION_1505",
               "history_role": "DEVELOPMENT_CALIBRATION", "new_market_requests": 0}
    return field, daily, summary

def fit(rows, core):
    validate_raw(rows[FIELDS].to_numpy(float))
    return optional_fit(rows, core, FIELDS)

def predict(core, model, base_values, raw_values):
    validate_raw(np.asarray(raw_values, float).reshape(1, -1))
    return optional_predict(core, model, base_values, raw_values, FIELDS)

def identity(rows, core):
    return optional_identity(rows, core, FIELDS)

def paths():
    own = [ROOT / "research/point_relative_price_optional_correction_v1.py",
           ROOT / "tests/test_point_relative_price_optional_correction_v1.py", COMPARISON, MASTER,
           OUT / "build_isolated_module.py", OUT / "prior_definition_and_function_addendum.json", OUT / "tests_receipt.json",
           ROOT / "research/point_optional_residual_model_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
           ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
           ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
           ROOT / "research/point_p02_exit_prediction_v1.py", ROOT / "research/point_d02_optional_correction_v1.py",
           CURRENT / "inputs/within_models.json", CURRENT / "inputs/config/within_cycle_exit.json",
           CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
           CURRENT / "inputs/candidate_features.parquet",
           OTHER / "protocol_source.json", OTHER / "source_result.json", OTHER / "两个ETF身份与原价格来源.json",
           OTHER / "两ETF原价与固定过去相对状态.parquet", OTHER / "实际旧用途与两个ETF观察身份.json",
           OTHER / "protocol.json", OTHER / "result.json", OTHER / "固定失败归因_信号费用与原叶.json",
           ROOT / "config/csi300_etf_rotation_alpha_v1.yaml", ROOT / "reports/backtest/csi300_etf_rotation_alpha_v1.json",
           ROOT / "config/510300_cross_etf_forced_flow_binary_screen_v1_candidates.yaml",
           ROOT / "config/510300_stress_transmission_and_exhaustion_atlas_v1.yaml",
           ROOT / "research/stress_transmission_and_exhaustion_atlas_v1.py",
           ROOT / "reports/research/510300_stress_transmission_and_exhaustion_atlas_v1.json",
           ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    return sorted(set(own))

def check():
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "相对价格金融协议改变。")
    for item in frozen["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结源改变：" + item["path"])
    return len(frozen["sources"])

def qualify():
    require(not (OUT / "source_protocol.json").exists(), "唯一源与设计资格已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["module_sha256"] == digest(ROOT / "research/point_relative_price_optional_correction_v1.py")
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_relative_price_optional_correction_v1.py"),
            "必要测试或版本不符。")
    sources = paths()
    protocol = {"at": now(), "study": STUDY, "technical_decision": "TECH.R143", "fields": FIELDS,
                "source_clock": "固定原510300/510500未复权价差，过去5/20原交易日，收盘15:00在15:05前；首版未认证。",
                "source_gate": "全部原3488日/1507状态保留；缺任何过去窗口原价保持未知；原115成熟月两列周期内设计标准SVD秩均2。",
                "rank_rule": "sqrt(w)*Dx的标准SVD，tol=max(shape)*float64机器精度*最大奇异值；不按结果选择容差。",
                "target_column_read": False, "new_model_fits": 0,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources]}
    write_json(OUT / "source_protocol.json", protocol, exclusive=True)
    field, daily, summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    raw = field[FIELDS].to_numpy(float)
    models = read(CURRENT / "inputs/within_models.json")["models"]
    months = []
    for record in models:
        if record["model"] is None:
            continue
        mask = states.cycle_id.isin(record["training_cycles"]).to_numpy()
        selected = states.loc[mask]
        require(len(selected) == record["training_rows"] and (selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(), "原成熟成员或时钟改变。")
        ids = selected.cycle_id.to_numpy(int)
        unique, counts = np.unique(ids, return_counts=True)
        count_map = dict(zip(unique, counts, strict=True))
        weights = np.asarray([1 / count_map[i] for i in ids], float)
        x = raw[mask]
        known = np.isfinite(x).all(axis=1)
        phi = np.zeros_like(x)
        if known.any():
            mean = np.average(x[known], axis=0, weights=weights[known])
            sd = np.sqrt(np.average((x[known] - mean)**2, axis=0, weights=weights[known]))
            z = np.clip((x[known] - mean) / np.where(sd > 1e-12, sd, 1.), -5., 5.)
            phi[known] = z - np.average(z, axis=0, weights=weights[known])
        dx = np.empty_like(phi)
        for cycle in unique:
            selected_cycle = ids == cycle
            dx[selected_cycle] = phi[selected_cycle] - np.average(phi[selected_cycle], axis=0, weights=weights[selected_cycle])
        matrix = np.sqrt(weights[:, None]) * dx
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
        rank = int((singular > tolerance).sum())
        months.append({"fit_origin": record["fit_origin"], "original_training_rows": len(selected),
                       "known_optional_training_rows": int(known.sum()), "rank": rank,
                       "minimum_singular_value": float(singular[-1]), "svd_tolerance": tolerance})
    require(len(months) == 115, "原115成熟月改变。")
    passed = all(item["rank"] == 2 for item in months)
    result = {"at": now(), "technical_decision": "TECH.R143",
              "status": "PASS_ORIGINAL_RELATIVE_PRICE_SOURCE_AND_TWO_COLUMN_DESIGN_NO_TARGET" if passed
                        else "REJECTED_FIXED_RELATIVE_PRICE_TWO_COLUMN_DESIGN_NOT_IDENTIFIABLE",
              "source_summary": summary, "ready_months": len(months),
              "identifiable_months": sum(item["rank"] == 2 for item in months), "monthly_support": months,
              "target_column_read": False, "new_model_fits": 0, "new_accounts": 0, "new_network_requests": 0,
              "design_gate_passed": passed, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    for item in protocol["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "资格执行中源改变。")
    table("资格_原相对价格日线字段", daily)
    table("资格_全部原状态相对价格字段", field)
    write_json(OUT / "source_and_design_qualification.json", result, exclusive=True)
    print(json.dumps({key:value for key,value in result.items() if key != "monthly_support"}, ensure_ascii=False))

def freeze():
    require(not (OUT / "freeze.json").exists(), "唯一相对价格金融方案已冻结，不重复。")
    qualified = read(OUT / "source_and_design_qualification.json")
    require(qualified["design_gate_passed"] and not qualified["target_column_read"], "源/设计资格失败或已读目标。")
    original = read(OUT / "source_protocol.json")
    for item in original["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "源资格后对象改变。")
    prior = read(OUT / "prior_definition_and_function_addendum.json")
    require(prior["other_branch_E71_failure_preserved"] and prior["different_function_bound"], "用途区别未成立。")
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R144", "result_decision": "TECH.R145",
                "candidate_configurations": 1, "fields": FIELDS, "hypothesis": prior["hypothesis"],
                "different_use": prior["different_use"], "source": original["source_clock"],
                "field": "(510300原close/过去5或20日close-1)-(510500原ts_close/相同原交易日close-1)；两固定窗口全价完整才有值，不是纯风格因果或资金流。",
                "members_and_fit": "原八项/尺度/截距固定，原全部行/目标/周期总权重1及142月115可用27未知、最近20/至少10周期100行和入场锁定不变；联合已知标准化clip正负5中心化，缺失修正不活动。设计与原目标减固定核心的残差各周期内中心化，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1/新截距0。缺原值精确回原核心；核心未知双方未知。",
                "prediction_gate": "完整原双期配对、周期等权原MSE均严格下降且5000原入场年块seed51030099改善95%下界均>0。",
                "periods": PERIODS,
                "economic_gate": "通过预测门后另冻结20万元252日BASE/STRESS双期完整账户：净CAGR/净夏普均高于A、实际净pB>1、标准净期望>0、DD<=10%、次数软目标；原50%上限/10pp带宽及全部风险限制保持。",
                "no_rescue": prior["no_rescue"], "other_branch_E71_failure_preserved": True,
                "normalizer_recomputation": "冻结前统一按原fit的rows[FIELDS].to_numpy(float)[known]选择和归约；保留1e-13，不修改原父模块。",
                "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
                "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
                "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    sources = paths() + [OUT / "source_protocol.json", OUT / "source_and_design_qualification.json",
                        OUT / "results/资格_原相对价格日线字段.parquet", OUT / "results/资格_全部原状态相对价格字段.parquet"]
    frozen = {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
              "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in sorted(set(sources))]}
    write_json(OUT / "freeze.json", frozen, exclusive=True)
    print("唯一固定相对价格金融协议已冻结；未读新标签或拟合。")

'''


def main():
    if DEST.exists():
        raise RuntimeError("隔离模块已经存在，不覆盖。")
    original = (ROOT / "research/point_d02_optional_correction_v1.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(original)
    names = {node.name: ast.get_source_segment(original, node) for node in tree.body
             if isinstance(node, ast.FunctionDef)}
    finance = names["run"] + "\n\n" + names["verify"]
    old_array = 'raw, w = rows.loc[known, FIELDS].to_numpy(float), rows.loc[known, "sample_weight"].to_numpy(float)'
    new_array = 'raw = rows[FIELDS].to_numpy(float)[known]\n                w = rows.sample_weight.to_numpy(float)[known]'
    if finance.count(old_array) != 1:
        raise RuntimeError("原尺度复算操作没有唯一匹配。")
    finance = finance.replace(old_array, new_array)
    replacements = {
        "原D02日历阶段字段": "原相对价格日线字段",
        "全部原状态D02可选字段": "全部原状态相对价格可选字段",
        "全部原状态D02固定预测": "全部原状态相对价格固定预测",
        "TECH.R137": "TECH.R145",
        "old_R94_direct_two_field_support_gate_passed": "other_branch_E71_failure_reversed",
        "complete_D02_all_member_field_admission": "complete_relative_price_all_member_field_admission",
        "SAME_R94_FIELDS": "ORIGINAL_ETF_RAW_PRICE_FIELDS",
        "D02": "RELATIVE_PRICE",
    }
    for old, new in replacements.items():
        finance = finance.replace(old, new)
    ending = '''

def main():
    parser = argparse.ArgumentParser(description="原相对价格固定两系数隔离研究")
    parser.add_argument("action", choices=["qualify", "freeze", "run", "verify"])
    arguments = parser.parse_args()
    {"qualify": qualify, "freeze": freeze, "run": run, "verify": verify}[arguments.action]()

if __name__ == "__main__":
    main()
'''
    code = PREFIX + finance + ending
    ast.parse(code)
    with DEST.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code)
    print("完整隔离模块已生成；原冻结模块保持，尚未读取目标或估计收益模型。")


if __name__ == "__main__":
    main()
