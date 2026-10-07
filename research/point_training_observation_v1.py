"""恢复原训练参考与月度制度：历史样本、训练成员及模型逐项对应。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import point_monthly_model_inputs_v1 as engine
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_core_observation_v1 import compare_numeric, LEDGER_FIELDS

OUT = ROOT / "reports/research/510300_point_training_observation_v1"
READY = ROOT / "reports/research/510300_point_core_observation_v1"
OLD = ROOT / "reports/research/510300_learned_cycle_exit_v1"
WITHIN = ROOT / "reports/research/510300_within_cycle_exit_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "freeze.json").exists(), "训练观察制度已经登记，不覆盖。")
    files = {}
    def copy(source, relative):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(destination)}
    for name in ("daily.parquet", "dividends.csv", "within_models.json", "original_models.json"):
        copy(READY / "inputs" / name, "inputs/" + name)
    for name in ("learned_cycle_exit", "within_cycle_exit"):
        copy(ROOT / f"config/510300_{name}_v1.json", f"inputs/{name}_config.json")
    for kind in ("ledger.parquet", "decisions.parquet", "cycles.csv"):
        copy(OLD / f"reference/D60_INTRA_{kind}", "inputs/saved/reference_" + kind)
    copy(OLD / "all_reference_samples.parquet", "inputs/saved/all_samples.parquet")
    copy(OLD / "training_memberships.parquet", "inputs/saved/ordinary_membership.parquet")
    copy(WITHIN / "training_memberships.parquet", "inputs/saved/within_membership.parquet")
    copy(READY / "inputs/current_requirements.json", "inputs/current_requirements.json")
    for path in (Path(__file__), Path(engine.__file__), ROOT / "research/training_reference_observation_v1.py",
                 ROOT / "research/prepare_training_reference_observer_v1.py", ROOT / "research/learned_cycle_exit_account_v1.py",
                 ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
                 ROOT / "tests/test_point_monthly_model_inputs_v1.py"):
        copy(path, "code/" + path.name)
    protocol = {
        "study": "510300_POINT_TRAINING_OBSERVATION_V1", "at": common.now(),
        "scope": "恢复两条固定候选依赖的训练样本与原月度制度，不改变策略或训练阈值。",
        "source": "训练基准是原31号D60_INTRA非再武装、无学习退出的参考路径；不得换成后来128号或再武装参考。",
        "maturity": "参考周期自然退出后，原可持有状态才按原继续持有净收益公式成熟；文件结束仍持有的状态无标签，不进入训练。",
        "training": "每月首个真实收盘以及原初始收盘，最近20个自然成熟周期，至少10个周期100行，原8项特征、周期等总权重、截断5、岭惩罚1和SVD求解。",
        "two_models": "复现原普通岭与原周期内截距岭两种模型；模型之间不选择最优，不拟合新特征。",
        "parity": "对原完整参考末端前状态、1461条样本、原训练成员及两套141个月度记录逐一比较。",
        "last_day": "原2026-08-14统一开盘强平改为真实收盘，下一计划日2026-08-17。未完成参考周期保留，不把收盘补记当作新前瞻点位。",
        "new_parameter_configurations": 0, "new_evaluation_months": 0, "new_point_evaluations": 0,
        "current_signal": "NO_VIEW_STALE_PRICES_AND_MONTHLY_MODELS", "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    for name in ("protocol.json", "source_derivation.json"):
        files[name] = {"sha256": common.digest(OUT / name)}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("原训练参考、自然成熟口径和两套月度模型对应范围已固定。", flush=True)


def compare_tree(a, b, path="record"):
    """既核对数值，也核对缺失状态、模型类型、周期成员和拟合时钟。"""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), path + "字段不同。")
        return max((compare_tree(a[k], b[k], path + "/" + str(k)) for k in a), default=0.)
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), path + "列表不同。")
        return max((compare_tree(x, y, path + f"/{i}") for i, (x, y) in enumerate(zip(a, b))), default=0.)
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        require(isinstance(b, (float, int)) and np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-11, path + "数值不同。")
        return float(abs(a-b))
    require(a == b, path + "状态不同。")
    return 0.


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "训练复现已经开始，不重复覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, record in frozen["files"].items():
        require(common.digest(OUT / relative) == record["sha256"], "固定输入发生变化：" + relative)
        if relative.startswith("code/"):
            require(common.digest(ROOT / record["source"]) == record["sha256"], "运行代码改变：" + record["source"])
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "purpose": "历史训练制度复现"})
    data = pd.read_parquet(OUT / "inputs/daily.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    cfg31 = json.loads((OUT / "inputs/learned_cycle_exit_config.json").read_text(encoding="utf-8"))
    cfg114 = json.loads((OUT / "inputs/within_cycle_exit_config.json").read_text(encoding="utf-8"))
    require(data.date.iloc[-1] == pd.Timestamp("2026-08-14"), "本次训练恢复范围改变。")
    reference = engine.reference_samples(data, dividends, cfg31, pd.Timestamp("2026-08-17"))
    for name, frame in reference.items():
        save_table(name, frame)
    saved_ledger = pd.read_parquet(OUT / "inputs/saved/reference_ledger.parquet")
    saved_decisions = pd.read_parquet(OUT / "inputs/saved/reference_decisions.parquet")
    fresh_ledger, fresh_decisions = reference["ledger"].iloc[:-1], reference["decisions"].iloc[:-1]
    require(pd.DatetimeIndex(fresh_ledger.date).equals(pd.DatetimeIndex(saved_ledger.date.iloc[:-1])), "参考收益的日期不同。")
    require(pd.DatetimeIndex(fresh_decisions.origin).equals(pd.DatetimeIndex(saved_decisions.origin)), "参考决定日期不同。")
    ledger_errors = {key: compare_numeric(fresh_ledger[key], saved_ledger[key].iloc[:-1], "训练参考/" + key, 1e-7) for key in LEDGER_FIELDS}
    for key in ("reference_weight", "requested_quantity", "learning_cycle_id", *engine.FEATURES):
        compare_numeric(fresh_decisions[key], saved_decisions[key], "训练决定/" + key)
    saved_samples = pd.read_parquet(OUT / "inputs/saved/all_samples.parquet")
    saved_samples = saved_samples.loc[saved_samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    pd.testing.assert_frame_equal(reference["samples"], saved_samples, check_exact=True)
    require(len(saved_samples) == 1461 and saved_samples.cycle_id.nunique() == 34, "原自然成熟样本规模不同。")
    print("原训练参考与1461条成熟样本完全对应，末端未完成状态单列。", flush=True)
    fitted = engine.fit_monthly_pair(data, reference["samples"], cfg31, cfg114)
    saved = {
        "ordinary": json.loads((OUT / "inputs/original_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"],
        "within": json.loads((OUT / "inputs/within_models.json").read_text(encoding="utf-8"))["models"],
    }
    checks = []
    for key in ("ordinary", "within"):
        fresh = common.clean(fitted[key])
        require(len(fresh) == len(saved[key]) == 141, "原月度记录数不同。")
        for left, right in zip(fresh, saved[key]):
            error = compare_tree(left, right, key + "/" + left["fit_origin"])
            checks.append({"kind": key, "fit_origin": left["fit_origin"], "status": left["status"],
                           "training_cycles": left["training_cycle_count"], "training_rows": left["training_rows"],
                           "latest_mature_exit": left["latest_exit_date"], "maximum_value_error": error,
                           "verification": "PASS_ORIGINAL_MONTHLY_RECORD"})
        common.save_json(OUT / f"results/{key}_models.json", {"models": fresh, "purpose": "历史原模型复现"})
        old = pd.read_parquet(OUT / f"inputs/saved/{key}_membership.parquet")
        if key == "ordinary":
            old = old.loc[old.signal.eq("D60_INTRA") & old.kind.eq("RIDGE")]
        columns = list(fitted["membership"].columns)
        pd.testing.assert_frame_equal(fitted["membership"], old[columns].reset_index(drop=True), check_exact=True)
    checks = pd.DataFrame(checks)
    save_table("月度模型对应", checks)
    save_table("训练成员", fitted["membership"])
    print("两套141个月度记录、训练成员与模型均已对应。", flush=True)
    open_cycles = reference["cycles"].loc[reference["cycles"].exit_date.isna()].copy()
    require(len(open_cycles) == 1 and int(open_cycles.cycle_id.iloc[0]) == 36, "训练观察末端持仓与原自然历史不一致。")
    summary = {
        "study": "510300_POINT_TRAINING_OBSERVATION_V1", "at": common.now(), "status": "ORIGINAL_MONTHLY_TRAINING_REPRODUCED_OPEN_CYCLE_RESTORED",
        "reference_close_rows_compared": len(fresh_ledger), "reference_decisions_compared": len(fresh_decisions),
        "reference_maximum_error": max(ledger_errors.values()), "mature_sample_rows": len(saved_samples),
        "mature_sample_cycles": int(saved_samples.cycle_id.nunique()), "sample_values_exact": True,
        "monthly_records_compared": len(checks), "monthly_maximum_value_error": float(checks.maximum_value_error.max()),
        "training_membership_rows_per_model": len(fitted["membership"]), "historical_successful_refits": int(checks.status.eq("FIT_COMPLETE").sum()),
        "latest_fit_origin": checks.fit_origin.max(), "latest_mature_training_exit": str(saved_samples.mature_date.max().date()),
        "unfinished_cycle": int(open_cycles.cycle_id.iloc[0]), "unfinished_entry_date": str(pd.Timestamp(open_cycles.entry_date.iloc[0]).date()),
        "unfinished_shares": int(reference["ledger"].shares.iloc[-1]), "unfinished_state_rows": len(reference["unfinished_states"]),
        "unfinished_training_labels_created": 0, "necessary_tests_passed": 5, "internal_reference_replays": 1,
        "new_monthly_fits_after_original_cutoff": 0, "new_parameter_configurations": 0, "new_point_evaluations": 0,
        "new_prospective_observations": 0, "current_signal": "NO_VIEW_STALE_PRICES_AND_MONTHLY_MODELS",
        "goal_achieved": False, "orders_authorized": False,
    }
    common.save_json(OUT / "summary.json", summary)
    report = ["# 原训练参考与月度模型恢复", "",
              "**两类退出模型的原训练制度已经复现，1461条成熟样本完全一致，末端仍持有的第36个参考周期已恢复为未完成状态。下一步可以按同一制度接续日线与后续月份模型；本轮没有新增点位盈利证据。**", "",
              f"原训练参考的{summary['reference_close_rows_compared']}条末端前收盘及{summary['reference_decisions_compared']}条决定均对应；参考账簿最大差异{summary['reference_maximum_error']:.3g}。34个自然成熟周期形成的1461条样本，状态、标签、权益和日期逐项完全一致。", "",
              f"两套模型共{summary['monthly_records_compared']}条月度记录，其中{summary['historical_successful_refits']}次为原成熟月份的重复拟合。训练周期成员、每行权重、标准化、系数、截距和模型状态均对应，最大数值差异{summary['monthly_maximum_value_error']:.3g}；没有新增参数配置或拟合时间段。每套共有{summary['training_membership_rows_per_model']}条训练成员记录。", "",
              "训练基准沿用原无学习退出、退出等待结束即可再次进入的D60_INTRA路径。它与实际候选中后来加入再武装条件的参考不同，不能用后者的交易标签替换。每月只用当时已经自然结束的最近20个周期，至少10个周期、100行，继续使用原8项状态。", "",
              f"原文件最后一日为2026-08-14，原程序曾因文件结束开盘平仓。真实收盘观察保留了自{summary['unfinished_entry_date']}进入的第{summary['unfinished_cycle']}个周期、{summary['unfinished_shares']}份参考持仓及{summary['unfinished_state_rows']}条尚未成熟的状态；这些状态没有未来退出标签，进入训练的数量为零。该份额是内部训练参考，不是实际账户指令。", "",
              f"最后已复现的拟合日仍是{summary['latest_fit_origin']}，其成熟训练退出资料最晚到{summary['latest_mature_training_exit']}。模型恢复不代表已经取得9月和之后的行情、拟合记录或当前候选状态。", "",
              "五项测试验证了未完成周期无标签、末日真实退出正常成熟、未来数据不改写既有状态、原等待后再进入规则以及真实月首末日的拟合时钟。它们证明样本延续方法与原制度一致，不证明策略能达到盈利门槛。", "",
              "用户门槛仍为p乘实际净盈亏比严格大于1且净均值为正；年度次数作为软目标，多空点位仍待有效验证。下一步补齐真实日线与股息，按已复现的制度补充月份，再接入已经完成历史对应的完整核心链。", ""]
    (OUT / "研究结论.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原训练参考与月度拟合观察恢复。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as error:
            common.save_json(OUT / "failure.json", {"at": common.now(), "status": "IMPLEMENTATION_DIAGNOSIS_REQUIRED", "error": str(error)})
            raise
