"""生成完整隔离模块，复用已验证的原成员比较流程，不修改模板或原策略。"""
import ast
from pathlib import Path

ROOT = Path.cwd()
DEST = ROOT / "research/point_employment_delivery_optional_correction_v1.py"
SOURCE = ROOT / "research/point_d02_optional_correction_v1.py"

PREFIX = '''"""原制造业就业/配送两问项的可选残差研究；原核心和全部成员保持。"""
from __future__ import annotations
import argparse
import copy
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.point_optional_residual_model_v1 import (
    design, system, fit as optional_fit, predict as optional_predict, identity as optional_identity, KIND)
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_volatility_unit_exit_v1 import model_at_entry

CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_employment_delivery_optional_correction_v1"
REGISTRY = ROOT / "reports/research/510300_point_macro_optional_correction_v1/source_fields_and_anchors.json"
OTHER = ROOT / "reports/research/510300_employment_delivery_cause_increment_v1"
FIELDS = ["MANUFACTURING_EMPLOYMENT_DI", "MANUFACTURING_DELIVERY_DI"]
STUDY = "510300_POINT_EMPLOYMENT_DELIVERY_OPTIONAL_CORRECTION_V1"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def table(name, frame):
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\\n")

def validate_raw(raw):
    require(raw.ndim == 2 and raw.shape[1] == 2, "就业配送固定两列。")
    require((np.isfinite(raw) | np.isnan(raw)).all(), "就业配送无穷值非法，不补值。")
    finite = np.isfinite(raw)
    require(((raw[finite] >= 0) & (raw[finite] <= 100)).all(), "原扩散指数必须为0至100，端点合法。")
    return np.isfinite(raw).all(axis=1)

def parse_current_month(raw, stat_month):
    require(re.fullmatch(r"\\d{4}-\\d{2}", stat_month) is not None, "统计月格式不同。")
    period = pd.Period(stat_month, freq="M")
    target = f"{period.year}年{period.month}月"
    soup = BeautifulSoup(raw, "html.parser")
    compact = lambda s: re.sub(r"\\s+", "", s)
    values, anchors = [], []
    for ti, tab in enumerate(soup.find_all("table")):
        rows = [[compact(c.get_text()) for c in tr.find_all(["td", "th"], recursive=False)]
                for tr in tab.find_all("tr")]
        headers = [r for r in rows if all(k in r for k in
                   ["生产", "新订单", "原材料库存", "从业人员", "供应商配送时间"])]
        if not headers:
            continue
        current = [r for r in rows if r and r[0] == target]
        require(len(headers) == len(current) == 1, "原制造业主表当月行或表头不唯一。")
        header, row = headers[0], current[0]
        offset = len(row) - len(header)
        require(offset in (0, 1, 2), "原主表头和行宽度不匹配。")
        columns = [header.index(k) + offset for k in ["从业人员", "供应商配送时间"]]
        try:
            pair = [float(row[k]) for k in columns]
        except (ValueError, IndexError) as exc:
            raise ValueError("就业配送当月原数值不可解析，不以旧月补值。") from exc
        require(validate_raw(np.asarray([pair])).all(), "原当月两项不是完整原值。")
        values.append(pair)
        anchors.append({"table_index": ti, "header": header, "current_month_row": row,
                        "value_columns": columns, "values": pair})
    require(values and all(v == values[0] for v in values), "当月就业配送主表缺失或重复表不一致。")
    return values[0], anchors

def monthly_values(records):
    monthly = pd.DataFrame(records).sort_values("stat_month").reset_index(drop=True)
    require(len(monthly) > 0 and monthly.stat_month.is_unique, "原月度记录为空或重复。")
    monthly["published_at"] = pd.to_datetime(monthly.published_at, utc=True).dt.tz_convert("Asia/Shanghai")
    monthly["known_at"] = pd.to_datetime(monthly.known_at, utc=True).dt.tz_convert("Asia/Shanghai")
    require(monthly.known_at.is_unique and monthly.known_at.is_monotonic_increasing, "报告可用钟必须递增唯一。")
    expected = monthly.published_at.dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    require(monthly.known_at.equals(expected), "原技术线延迟到公布日23:59:59，不借用另一任务的较早钟。")
    monthly["auxiliary_available"] = validate_raw(monthly[FIELDS].to_numpy(float))
    return monthly

def align(states, monthly):
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原自然身份重复。")
    clocks = pd.DatetimeIndex(monthly.known_at).as_unit("ns").asi8
    rows = []
    for state in states.itertuples(index=False):
        at = pd.Timestamp(state.origin).normalize().tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        index = int(np.searchsorted(clocks, at.value, side="right") - 1)
        source = monthly.iloc[index] if index >= 0 else None
        values = [float(source[k]) for k in FIELDS] if source is not None else [np.nan, np.nan]
        available = bool(validate_raw(np.asarray([values]))[0])
        rows.append({"cycle_id": int(state.cycle_id), "origin_index": int(state.origin_index), "origin": state.origin,
                     "origin_at": at, "stat_month": source.stat_month if source is not None else None,
                     "known_at": source.known_at if source is not None else pd.NaT,
                     "method_group": source.method_group if source is not None else None,
                     FIELDS[0]: values[0], FIELDS[1]: values[1], "auxiliary_available": available,
                     "availability_status": "KNOWN_OPTIONAL_INPUT" if available else "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK"})
    return pd.DataFrame(rows)

def load_fields():
    registry = read(REGISTRY)
    require(len(registry) == 140 and [r["stat_month"] for r in registry]
            == pd.period_range("2015-01", "2026-08", freq="M").astype(str).tolist(), "原140月母集变化。")
    old = pd.read_parquet(OTHER / "原公布就业与配送两字段.parquet").set_index("stat_month")
    require(len(old) == 132, "另一任务132原月源变化。")
    records, all_anchors = [], []
    for r in registry:
        raw = (ROOT / r["raw_path"]).read_bytes()
        require(digest(ROOT / r["raw_path"]) == r["raw_sha256"], "原HTML摘要变化：" + r["stat_month"])
        values, anchors = parse_current_month(raw, r["stat_month"])
        record = {k: r[k] for k in ["stat_month", "published_at", "known_at", "method_group", "source_url", "raw_path", "raw_sha256"]}
        record.update(dict(zip(FIELDS, values)))
        if r["stat_month"] in old.index:
            prior = old.loc[r["stat_month"]]
            require(prior.raw_sha256 == r["raw_sha256"] and prior.method_group == r["method_group"]
                    and values == [float(prior["制造业从业人员原扩散指数"]), float(prior["制造业供应商配送时间原扩散指数"])],
                    "132旧原值/源/方法组不一致。")
        records.append(record)
        all_anchors.append({"stat_month": r["stat_month"], "raw_path": r["raw_path"],
                            "raw_sha256": r["raw_sha256"], "anchors": anchors})
    monthly = monthly_values(records)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    require(len(states) == 1507 and pd.Timestamp(states.origin.max()) <= pd.Timestamp("2026-09-24"), "原自然母集或末端改变。")
    field = align(states, monthly)
    require(np.array_equal(states[["cycle_id", "origin_index"]], field[["cycle_id", "origin_index"]]), "原自然身份改变。")
    summary = {"status": "PASS_LOCAL_ORIGINAL_EMPLOYMENT_DELIVERY_FIELDS_OPTIONAL_FUNCTION_ONLY",
               "monthly_reports": len(monthly), "prior_132_values_identical": True,
               "additional_local_2026_months_parsed": 8, "original_state_rows": len(states),
               "optional_known_rows": int(field.auxiliary_available.sum()),
               "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
               "method_group_counts": monthly.method_group.value_counts().to_dict(),
               "physical_first_vintage_verified": False, "complete_employment_delivery_all_member_field_admission": False,
               "source_clock": "INHERITED_REPORT_DAY_235959_ORIGINAL_CLOSE_1505",
               "history_role": "DEVELOPMENT_CALIBRATION", "new_market_requests": 0}
    return field, monthly, summary

def fit(rows, core):
    validate_raw(rows[FIELDS].to_numpy(float))
    return optional_fit(rows, core, FIELDS)

def predict(core, model, base_values, raw_values):
    validate_raw(np.asarray(raw_values, float).reshape(1, -1))
    return optional_predict(core, model, base_values, raw_values, FIELDS)

def identity(rows, core):
    return optional_identity(rows, core, FIELDS)

def paths():
    own = [Path(__file__), ROOT / "tests/test_point_employment_delivery_optional_correction_v1.py", REGISTRY,
           OUT / "prior_definition_and_function_addendum.json", OUT / "tests_receipt.json",
           ROOT / "research/point_optional_residual_model_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
           ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
           ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
           ROOT / "research/point_p02_exit_prediction_v1.py", ROOT / "research/point_d02_optional_correction_v1.py",
           OUT / "build_isolated_module.py", OTHER / "protocol_source.json", OTHER / "source_result.json",
           OTHER / "实际旧用途与原就业配送问项核对.json", OTHER / "原公布就业与配送两字段.parquet",
           OTHER / "protocol.json", OTHER / "result.json",
           CURRENT / "inputs/within_models.json", CURRENT / "inputs/config/within_cycle_exit.json",
           CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
           CURRENT / "inputs/candidate_features.parquet",
           ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    own += [ROOT / r["raw_path"] for r in read(REGISTRY)]
    return sorted(set(own))

def check():
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "就业配送金融协议改变。")
    for item in frozen["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结源改变：" + item["path"])
    return len(frozen["sources"])

def qualify():
    require(not (OUT / "source_protocol.json").exists(), "唯一源与设计资格已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_employment_delivery_optional_correction_v1.py"), "必要测试或版本不符。")
    sources = paths()
    protocol = {"at": now(), "study": STUDY, "technical_decision": "TECH.R140",
                "fields": FIELDS, "source_clock": "原技术线140月日志，公布日23:59:59，原点15:05；原首版未认证。",
                "source_gate": "原140连续月/132旧值与源一致；全部原1507成员保留，原缺失保持；原115成熟月两列周期内设计标准SVD秩均2。",
                "rank_rule": "对sqrt(w)*Dx作SVD；tol=max(shape)*float64机器精度*最大奇异值。原权重、clip正负5及中心化不变，不依据测量选择容差。",
                "target_column_read": False, "new_model_fits": 0,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources]}
    write_json(OUT / "source_protocol.json", protocol, exclusive=True)
    field, monthly, summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    raw = field[FIELDS].to_numpy(float)
    models = read(CURRENT / "inputs/within_models.json")["models"]
    months = []
    for r in models:
        if r["model"] is None:
            continue
        mask = states.cycle_id.isin(r["training_cycles"]).to_numpy()
        selected = states.loc[mask]
        require(len(selected) == r["training_rows"] and (selected.exit_index <= r["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(r["fit_origin"])).all(), "原成熟成员或时钟改变。")
        ids = selected.cycle_id.to_numpy(int)
        unique, counts = np.unique(ids, return_counts=True)
        count_map = dict(zip(unique, counts))
        w = np.asarray([1 / count_map[i] for i in ids], float)
        x = raw[mask]
        known = np.isfinite(x).all(axis=1)
        phi = np.zeros_like(x)
        if known.any():
            mean = np.average(x[known], axis=0, weights=w[known])
            sd = np.sqrt(np.average((x[known] - mean)**2, axis=0, weights=w[known]))
            z = np.clip((x[known] - mean) / np.where(sd > 1e-12, sd, 1.), -5., 5.)
            phi[known] = z - np.average(z, axis=0, weights=w[known])
        dx = np.empty_like(phi)
        for cycle in unique:
            m = ids == cycle
            dx[m] = phi[m] - np.average(phi[m], axis=0, weights=w[m])
        a = np.sqrt(w[:, None]) * dx
        singular = np.linalg.svd(a, compute_uv=False)
        tolerance = float(max(a.shape) * np.finfo(float).eps * singular[0])
        rank = int((singular > tolerance).sum())
        months.append({"fit_origin": r["fit_origin"], "original_training_rows": len(selected),
                       "known_rows": int(known.sum()), "unknown_original_rows_retained": int((~known).sum()),
                       "weighted_design_singular_values": singular.tolist(), "standard_tolerance": tolerance, "rank": rank})
    require(len(models) == 142 and len(months) == 115, "原月状态数量改变。")
    passed = all(x["rank"] == 2 for x in months)
    result = {"at": now(), "technical_decision": "TECH.R140",
              "status": "PASS_ORIGINAL_EMPLOYMENT_DELIVERY_SOURCE_AND_TWO_COLUMN_DESIGN_NO_TARGET" if passed
                        else "REJECTED_FIXED_EMPLOYMENT_DELIVERY_TWO_COLUMN_DESIGN_NOT_IDENTIFIABLE",
              "source_summary": summary, "ready_months": len(months),
              "identifiable_months": sum(x["rank"] == 2 for x in months), "monthly_support": months,
              "target_column_read": False, "new_model_fits": 0, "new_accounts": 0, "new_network_requests": 0,
              "design_gate_passed": passed, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    for item in protocol["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "资格执行中源改变。")
    table("资格_原月度就业配送字段", monthly)
    table("资格_全部原状态就业配送字段", field)
    write_json(OUT / "source_and_design_qualification.json", result, exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k != "monthly_support"}, ensure_ascii=False))

def freeze():
    require(not (OUT / "freeze.json").exists(), "唯一就业配送金融方案已冻结，不重复。")
    qualified = read(OUT / "source_and_design_qualification.json")
    require(qualified["design_gate_passed"] and not qualified["target_column_read"], "源/设计资格失败或已读目标。")
    original = read(OUT / "source_protocol.json")
    for item in original["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "源资格后对象改变。")
    prior = read(OUT / "prior_definition_and_function_addendum.json")
    require(prior["other_branch_E65_failure_preserved"] and prior["different_function_bound"], "用途区别未成立。")
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R141", "result_decision": "TECH.R142",
                "candidate_configurations": 1, "fields": FIELDS, "hypothesis": prior["hypothesis"],
                "different_use": prior["different_use"], "source": original["source_clock"],
                "field": "两官方原季调扩散指数水平，0至100；配送高代表更快，不是交货天数、确定需求原因或市场预期差。不加差分、方向阈值或交互。",
                "method_group_assumption": "原分类/样本方法组全部保留；不同组原DI水平同尺度是开发假设，未证明总体完全可比，不按结果分组营救。",
                "members_and_fit": "原八项/尺度/截距固定，原全部行/目标/周期总权重1及142月115可用27未知、最近20/至少10周期100行和入场锁定不变；联合已知标准化clip正负5中心化，缺失修正不活动。设计与原目标减固定核心的残差各周期内中心化，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1/新截距0。缺原值精确回原核心；核心未知双方未知。",
                "prediction_gate": "完整原双期配对、周期等权原MSE均严格下降且5000原入场年块seed51030099改善95%下界均>0。",
                "periods": PERIODS, "economic_gate": "过预测门才另冻结完整20万252日BASE/STRESS双期账户：净CAGR/夏普均高于A，实际净pB>1、标准期望pB-q>0、回撤<=10%，次数软目标。",
                "no_rescue": "不换字段、窗口、符号、维度、alpha、标签、缺失掩码、方法组、时期、成员/权重或费用，不从本批十一失败选单期或混合。",
                "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
                "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
                "new_return_labels": 0, "new_market_requests": 0, "goal_achieved": False, "orders_authorized": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    sources = paths() + [OUT / "source_protocol.json", OUT / "source_and_design_qualification.json"]
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
               "sources": [{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in sorted(set(sources))]}, exclusive=True)
    print("就业配送唯一两系数方案已冻结，另一分支E65失败保留。")

'''

def main():
    if DEST.exists():
        raise ValueError("隔离模块已存在，不覆盖。")
    original = SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(original)
    names = {node.name: ast.get_source_segment(original, node) for node in tree.body
             if isinstance(node, ast.FunctionDef)}
    finance = names["run"] + "\n\n" + names["verify"]
    replacements = {
        "原D02日历阶段字段": "原月度就业配送字段",
        "全部原状态D02可选字段": "全部原状态就业配送可选字段",
        "全部原状态D02固定预测": "全部原状态就业配送固定预测",
        "TECH.R137": "TECH.R142",
        "old_R94_direct_two_field_support_gate_passed": "other_branch_E65_failure_reversed",
        "complete_D02_all_member_field_admission": "complete_employment_delivery_all_member_field_admission",
        '"daily_rows": len(daily)': '"monthly_reports": len(daily)',
        "SAME_R94_FIELDS": "ORIGINAL_LOCAL_FIELDS",
        "D02": "EMPLOYMENT_DELIVERY",
    }
    for old, new in replacements.items():
        finance = finance.replace(old, new)
    end = '''

def main():
    parser = argparse.ArgumentParser(description="原就业配送固定两系数隔离研究")
    parser.add_argument("action", choices=["qualify", "freeze", "run", "verify"])
    arguments = parser.parse_args()
    {"qualify": qualify, "freeze": freeze, "run": run, "verify": verify}[arguments.action]()

if __name__ == "__main__":
    main()
'''
    code = PREFIX + finance + end
    ast.parse(code)
    with DEST.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code)
    print("完整隔离模块已生成，原D02模块保持，未读取目标或估计收益模型。")

if __name__ == "__main__":
    main()
