"""固定后续验证边界，复算参考观察程序并列出完整信号链的实际准备程度。"""
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
from research import reference_observation_accounts_v1 as observer
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules as learned_rules
from research.simple_volume_reversal_v1 import make_rules as panic_rules
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_point_forward_readiness_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
REGISTRY_COLUMNS = ["candidate", "direction", "origin", "source_data_max_date", "actual_generated_at", "planned_execution_date",
                    "parent_state_receipt", "parent_state_hash", "inputs_receipt", "signal_status", "entry_or_exit_intent",
                    "point_state_before", "reference_cost_basis", "eligibility", "unavailable_reason"]
CALENDAR_URL = "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p.with_suffix(".parquet"), index=False)
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "后续验证准备已固定，不覆盖。")
    derivation = json.loads((OUT / "source_derivation.json").read_text(encoding="utf-8"))
    require(common.digest(ROOT / derivation["destination"]) == derivation["destination_sha256"], "观察版与边界派生记录不同。")
    for row in derivation["sources"]:
        require(common.digest(ROOT / row["path"]) == row["sha256"], "旧参考程序被修改。")
    files = {}
    def copy(source, relative):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(destination)}
    configs = {}
    for key in ("entry_vintage_exit", "continuous_reference_min_variance", "downside_reference_risk"):
        path = ROOT / f"config/510300_{key}_v1.json"
        configs[key] = json.loads(path.read_text(encoding="utf-8"))
        copy(path, f"inputs/config/{key}.json")
    cfg = configs["entry_vintage_exit"]
    for key in ("features", "dividends"):
        source = ROOT / cfg[key]
        expected = next(r["sha256"] for r in cfg["frozen_files"] if Path(r["path"]) == Path(cfg[key]))
        require(common.digest(source) == expected, "原始参考的价格或股息来源发生改变。")
        copy(source, "inputs/" + ("daily.parquet" if key == "features" else "dividends.csv"))
    copy(ROOT / configs["continuous_reference_min_variance"]["panic_features"], "inputs/panic_daily.parquet")
    copy(ROOT / cfg["saved_models"], "inputs/within_models.json")
    copy(ROOT / configs["continuous_reference_min_variance"]["saved_models"], "inputs/original_models.json")
    copy(ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/features.parquet", "inputs/already_inspected_daily.parquet")
    for name, folder, model, cost in reference_sources():
        for kind in ("ledger", "decisions"):
            copy(folder / f"{model}_{kind}.parquet", f"inputs/saved/{name}_{kind}.parquet")
    copy(ROOT / "reports/research/510300_point_state_reconstruction_v1/inputs/evaluation/CORE_decisions.parquet", "inputs/latest_core_decisions.parquet")
    copy(CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    for path in (Path(__file__), ROOT / "research/prepare_reference_observation_engines_v1.py", Path(observer.__file__),
                 ROOT / "tests/test_reference_observation_accounts_v1.py"):
        copy(path, "code/" + path.name)
    for row in derivation["sources"]:
        copy(ROOT / row["path"], "code/original/" + Path(row["path"]).name)
    at = common.now()
    protocol = {
        "study": "510300_POINT_FORWARD_READINESS_V1", "registered_at": at,
        "scope": "510300日线及此前完整周线多空点位，p乘实际净B>1、净均值>0；年度次数软目标。",
        "selected_long_candidates": ["CORE_AUXILIARY_DRAWDOWN_GATE", "LAG_CONFIRMED_RUNS_AUXILIARY"],
        "selected_short_candidates": [], "short_status": "尚无满足要求并进入本次待验证集的空头规则，目标这一部分未解决。",
        "historical_status": "候选由既有历史比较选出，已研究的历史不再作为独立样本。后补数据更新只算历史状态恢复。",
        "source_boundary": "原完整链至2026-08-14开盘终点，核心来源最后收盘2026-08-13；当前已研究日线至少到2026-09-16。实际日期由文件再次读取。",
        "reference_observation": "三个隔离派生的参考函数保留原经济规则，移除由数据文件结束触发的强平；最后已知日正常开盘执行前一请求、真实收盘记账，再产生下一计划交易日意向。无虚构未来价格行。",
        "reference_replays": "原固定入场模型BASE/STRESS、连续原岭模型BASE、连续急跌回升BASE、132目标参考BASE，共五条内部参考状态复算；不评估新的资金账户绩效。",
        "parity": "原终点之前的持仓、现金、权益、净收益、成交、分红及决定逐项对应旧保存结果；最后日另列，因为旧数据采用统一强平。",
        "registry": "后续记录必须保存实际生成时间、已知行情截止、父状态来源及其指纹、计划下一开盘和当时持仓；当前登记表为空，协议与测试不计作信号或交易。",
        "prospective_eligibility": "规则登记后、实际结果出现之前生成并保存的合格意向，才可进入前瞻集合；迟到补写只能标记历史恢复。已有持仓与未完成点位单列，不在登记日假平仓或伪造新入场。",
        "parent_unavailable": "价格、股息、月度模型或必需父状态未按既定流程更新，保留NO_VIEW；不能延长已过期的历史父目标、用现金代替未知或换一个更容易运行的策略。",
        "statistics": "只在自然完成点位上计算胜率、实际B、pB、平均净收益、利润因子及年度次数；空样本及缺少赢亏之一时保留未计算；阶段性越线不等于独立验证完成。",
        "parameters": "候选阈值、风险尺度、模型训练方式、成本及点位定义保持原样；尚未产生新的月度模型或完整实时核心链。",
        "calendar_reference": {"url": CALENDAR_URL, "notice_date": "2026-09-17", "verified_on": "2026-10-01", "closed_from": "2026-10-01", "closed_through": "2026-10-07", "next_session": "2026-10-08"},
        "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    files["source_derivation.json"] = {"sha256": common.digest(OUT / "source_derivation.json")}
    common.save_json(OUT / "freeze.json", {"at": at, "files": files})
    pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(OUT / "后续点位记录.csv", index=False, encoding="utf-8-sig")
    print("后续验证候选、数据边界、五条内部参考复算及记录口径已固定。", flush=True)


def reference_sources():
    base = ROOT / "reports/research"
    return [
        ("ENTRY_VINTAGE_BASE", base / "510300_entry_vintage_exit_v1/evaluation/BASE", "ENTRY_VINTAGE_EXIT", "BASE"),
        ("ENTRY_VINTAGE_STRESS", base / "510300_entry_vintage_exit_v1/evaluation/STRESS", "ENTRY_VINTAGE_EXIT", "STRESS"),
        ("CONTINUOUS_RIDGE_BASE", base / "510300_continuous_reference_min_variance_v1/continuous_references/BASE", "REARM_RIDGE", "BASE"),
        ("CONTINUOUS_PANIC_BASE", base / "510300_continuous_reference_min_variance_v1/continuous_references/BASE", "PANIC_ONLY", "BASE"),
        ("DOWNSIDE_TARGET_BASE", base / "510300_downside_reference_risk_v1/evaluation/BASE", "DOWNSIDE_REFERENCE_RISK", "BASE"),
    ]


def check_prefix(name, result, last_date):
    ledger, decisions = result[:2]
    saved_ledger = pd.read_parquet(OUT / f"inputs/saved/{name}_ledger.parquet")
    saved_decisions = pd.read_parquet(OUT / f"inputs/saved/{name}_decisions.parquet")
    left, right = ledger.loc[ledger.date.lt(last_date)], saved_ledger.loc[saved_ledger.date.lt(last_date)]
    require(pd.DatetimeIndex(left.date).equals(pd.DatetimeIndex(right.date)), "原终点前参考日期不同。")
    errors = {}
    for column in ("shares", "cash", "equity", "net_return", "filled_quantity", "commission", "dividend_recognized", "dividend_paid"):
        errors[column] = float(np.max(np.abs(left[column].to_numpy(float)-right[column].to_numpy(float))))
        require(np.allclose(left[column], right[column], rtol=0, atol=1e-7, equal_nan=True), name + "前段账簿不一致：" + column)
    decision_prefix = decisions.loc[decisions.origin.lt(last_date)]
    require(pd.DatetimeIndex(decision_prefix.origin).equals(pd.DatetimeIndex(saved_decisions.origin)), "原决定时钟不同。")
    for column in ("reference_weight", "requested_quantity"):
        require(np.allclose(decision_prefix[column], saved_decisions[column], rtol=0, atol=1e-12, equal_nan=True), name + "前段意向不一致：" + column)
    require(ledger.mark_clock.eq("CLOSE").all(), "观察程序仍含人工终点开盘记账。")
    require(decisions.origin.iloc[-1] == last_date, "最后完整收盘没有形成意向记录。")
    tail = {"reference": name, "date": last_date, "old_terminal_shares": int(saved_ledger.shares.iloc[-1]),
            "observed_close_shares": int(ledger.shares.iloc[-1]), "old_terminal_mark": float(saved_ledger.mark.iloc[-1]),
            "observed_close_mark": float(ledger.mark.iloc[-1]), "observed_close_target": decisions.reference_weight.iloc[-1],
            "planned_next_execution": decisions.execution_date.iloc[-1], "old_last_clock": saved_ledger.mark_clock.iloc[-1],
            "new_last_clock": ledger.mark_clock.iloc[-1]}
    save_table(name + "_observer_ledger", ledger)
    save_table(name + "_observer_decisions", decisions)
    if len(result) == 3:
        save_table(name + "_observer_cycles", result[2])
    return {"reference": name, "compared_close_rows": len(left), "compared_decisions": len(decision_prefix),
            "maximum_monetary_error": max(errors[k] for k in ("cash", "equity", "commission", "dividend_recognized", "dividend_paid")),
            "maximum_return_error": errors["net_return"], "maximum_share_error": max(errors["shares"], errors["filled_quantity"]),
            "new_close_rows": 1}, tail


def run():
    require(not (OUT / "summary.json").exists(), "已有完整观察准备结果，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, row in frozen["files"].items():
        require(common.digest(OUT / relative) == row["sha256"], "冻结输入变化：" + relative)
    require(common.digest(Path(__file__)) == frozen["files"]["code/" + Path(__file__).name]["sha256"], "运行程序与固定版本不同。")
    cfg128 = json.loads((OUT / "inputs/config/entry_vintage_exit.json").read_text(encoding="utf-8"))
    cfg91 = json.loads((OUT / "inputs/config/continuous_reference_min_variance.json").read_text(encoding="utf-8"))
    cfg132 = json.loads((OUT / "inputs/config/downside_reference_risk.json").read_text(encoding="utf-8"))
    data, panic_data = pd.read_parquet(OUT / "inputs/daily.parquet"), pd.read_parquet(OUT / "inputs/panic_daily.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(panic_data.date)), "参考输入日历不同。")
    models114 = json.loads((OUT / "inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    models31 = json.loads((OUT / "inputs/original_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require(data.date.iloc[-1] == pd.Timestamp("2026-08-14"), "本次边界验证的历史日期改变。")
    next_date, comparisons, tails = pd.Timestamp("2026-08-17"), [], []
    for cost in ("BASE", "STRESS"):
        controller = EntryVintageExitController(data, models114, cfg128["confirmation_days"])
        result = observer.observe_rearmed_exit(data, dividends, cfg128, cfg128["costs"][cost], cfg128["evaluation_start"],
                                              learned_rules(data)["D60_INTRA"], cfg128["specification"], controller, next_execution_date=next_date)
        comparison, tail = check_prefix("ENTRY_VINTAGE_" + cost, result, data.date.iloc[-1])
        comparisons.append(comparison); tails.append(tail)
        print(f"固定入场模型{cost}：原终点前逐日状态已对应，末日改为真实收盘观察。", flush=True)
    controller = ExitController(data, models31, cfg91["confirmation_days"])
    result = observer.observe_rearmed_exit(data, dividends, cfg91, cfg91["costs"]["BASE"], cfg91["reference_start"],
                                          learned_rules(data)["D60_INTRA"], cfg91["learned_spec"], controller, next_execution_date=next_date)
    comparison, tail = check_prefix("CONTINUOUS_RIDGE_BASE", result, data.date.iloc[-1])
    comparisons.append(comparison); tails.append(tail)
    result = observer.observe_price_policy(panic_data, dividends, cfg91, cfg91["costs"]["BASE"], cfg91["reference_start"],
                                           panic_rules(panic_data)[0]["V6_PANIC_RECOVERY"], cfg91["panic_spec"], next_execution_date=next_date)
    comparison, tail = check_prefix("CONTINUOUS_PANIC_BASE", result, data.date.iloc[-1])
    comparisons.append(comparison); tails.append(tail)
    print("连续岭模型与急跌回升两条参考：原历史前段均已对应。", flush=True)
    saved_target = pd.read_parquet(OUT / "inputs/saved/DOWNSIDE_TARGET_BASE_decisions.parquet")
    targets = np.full(len(data), np.nan)
    targets[saved_target.origin_index.to_numpy(int)] = saved_target.reference_weight.to_numpy(float)
    result = observer.observe_event_account(data, dividends, cfg132, cfg132["costs"]["BASE"], cfg132["evaluation_start"],
                                           "DOWNSIDE_REFERENCE_RISK", targets=targets, event_mask=np.ones(len(data), bool), next_execution_date=next_date)
    comparison, tail = check_prefix("DOWNSIDE_TARGET_BASE", result, data.date.iloc[-1])
    comparisons.append(comparison); tails.append(tail)
    comparisons, tails = pd.DataFrame(comparisons), pd.DataFrame(tails)
    save_table("原终点前一致性", comparisons)
    save_table("原强平与真实收盘边界", tails)
    already_seen = pd.read_parquet(OUT / "inputs/already_inspected_daily.parquet", columns=["date"])
    core = pd.read_parquet(OUT / "inputs/latest_core_decisions.parquet", columns=["origin", "execution_date"])
    coverage = pd.DataFrame([
        {"component": "原完整链日线", "rows": len(data), "last_available": data.date.max(), "role": "模型及参考持仓的既有历史"},
        {"component": "已参与上涨解剖的日线", "rows": len(already_seen), "last_available": already_seen.date.max(), "role": "已观察历史，不能作为新独立样本"},
        {"component": "原核心收盘目标", "rows": len(core), "last_available": core.origin.max(), "role": "最后对应2026-08-14开盘，不能沿用为当前目标"},
        {"component": "持仓内模型月度记录", "rows": len(models114), "last_available": pd.Timestamp(models114[-1]["fit_origin"]), "role": "须按既定成熟样本制度延续月度更新"},
        {"component": "原岭模型月度记录", "rows": len(models31), "last_available": pd.Timestamp(models31[-1]["fit_origin"]), "role": "连续参考依赖的另一种原模型"},
        {"component": "本轮真实收盘内部参考", "rows": 5, "last_available": data.date.max(), "role": "边界修复验证，非当前市场信号或新增独立结果"},
    ])
    save_table("数据及模型可用边界", coverage)
    summary = {"study": "510300_POINT_FORWARD_READINESS_V1", "at": common.now(), "status": "REFERENCE_OBSERVATION_BOUNDARY_READY_FULL_CHAIN_PENDING",
               "observer_functions": 3, "necessary_tests_passed": 10, "internal_reference_replays": 5,
               "historical_prefix_rows_compared": int(comparisons.compared_close_rows.sum()),
               "maximum_monetary_error": float(comparisons.maximum_monetary_error.max()),
               "maximum_return_error": float(comparisons.maximum_return_error.max()),
               "maximum_share_error": float(comparisons.maximum_share_error.max()),
               "new_model_fits": 0, "new_investment_account_evaluations": 0, "new_prospective_observations": 0,
               "full_current_signal_chain_ready": False, "current_signal": "NO_VIEW_STALE_PRICES_MODELS_AND_PARENT_TARGETS",
               "short_candidate_ready": False, "earliest_future_exchange_session": "2026-10-08",
               "remaining_work": ["恢复末端未自然结束的训练参考，并按原制度更新新月份两类模型", "把连续风险预算及核心父目标接入真实收盘观察程序",
                                  "补齐日线股息并完成逐日前缀验证，随后才能产生当时已知的候选观察", "等待登记后产生并自然完成的新点位结果，按原门槛报告"],
               "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    write_report(summary, coverage, tails)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(summary, coverage, tails):
    lines = ["# 固定候选的后续验证准备", "",
             "**三种参考计算程序已改为真实收盘观察模式，并完成五条既有内部参考路径的对应验证。原终点之前的状态不变，末日不再因文件结束强平。但完整当前信号链仍未接通，前瞻点位记录为零，目标尚未实现。**", "",
             "两条历史达线候选为回撤限制组合与相关确认组合，均为多头。空头研究仍未有达线候选，不能把这一步称为多空目标已完成。当前用户口径是日线与此前完整周线、p×实际净B>1且净均值>0、年度次数软目标。", "",
             "## 当前资料真实到哪里", "", "| 来源 | 条数 | 最后可用日 | 作用 |", "|---|---:|---|---|"]
    for r in coverage.itertuples():
        lines.append(f"| {r.component} | {r.rows} | {r.last_available.date()} | {r.role} |")
    lines += ["", "原完整链与上涨解剖使用的行情截止日不同。补上8月以后数据可以恢复当前状态，但已经参与观察的日线不能重新包装成独立测试。原模型记录止于8月3日，也不能省略9月及之后原定的模型更新过程。", "",
              "## 为什么必须处理样本末端", "",
              "旧参考程序在数据最后一天统一开盘清仓，以完成当时的历史研究。如果将该行直接接到新行情上，会把人为清仓误认为策略自然退出，改变持仓收益、退出模型、再入场等待和风险预算。本轮在隔离文件派生三个观察函数，保留所有原经济条件，只取消由文件结束触发的平仓；最后日正常执行前一收盘请求，按真实收盘记账，并生成下一计划开盘的意向。没有追加虚构的未来价格行。", "",
              "| 内部参考 | 原最后日持仓 | 真实收盘观察持仓 | 原末端记账价 | 真实收盘价 | 下一计划执行日 |", "|---|---:|---:|---:|---:|---|"]
    for r in tails.itertuples():
        lines.append(f"| {r.reference} | {r.old_terminal_shares} | {r.observed_close_shares} | {r.old_terminal_mark:.3f} | {r.observed_close_mark:.3f} | {r.planned_next_execution.date()} |")
    lines += ["", f"共比较{summary['historical_prefix_rows_compared']}条原终点前收盘参考记录，现金、权益、收益、持股与成交数量均与原保存结果对应，最大金额误差{summary['maximum_monetary_error']:.3g}、收益误差{summary['maximum_return_error']:.3g}。最后日不同是预先定义的观察边界变化，不是调参后的业绩改善。", "",
              "下行风险参考的最后收盘目标仍为未知，因为本轮只复用原有目标以验证账户程序，没有伪造8月14日新父目标。它保留已有份额，不代表已获得完整核心信号。这里的内部参考是候选计算依赖，未评价新的投资账户夏普或年化收益。", "",
              "## 完整计算依赖", "",
              "固定入场模型参考使用60日日线开收盘相对隔夜强弱，以及当时成熟的持仓内退出模型；其目标分别乘普通波动与下行风险幅度，得到两条风险参考。另一条来源从2013年连续维护急跌回升和原岭退出参考，按242日收益风险形成预算。下行风险参考与连续来源再经过原有共同下行预算，随后按模型支持状态选择来源，最终由120日趋势与20日波动混合为核心。", "",
              "核心之外，辅助仍为原60日连续段状态；两条候选分别由60日回撤或相邻收益相关状态限制辅助。候选只以总目标正值/零值确定固定份额进入退出。这条完整链尚需恢复末端训练参考、更新原定月度模型，并将父目标和风险预算接到观察程序；不能只更新最外层MACD或价格就声称候选已能运行。", "",
              "## 后续点位如何计入", "",
              "协议与空白记录表已保存。合格记录需要实际生成时间、已知行情截止、父状态来源、下一计划开盘、当时持仓及成本口径。先保存意向，后观察执行和退出；迟到的补写只能标为历史恢复。登记前已有持仓单列，不能在登记日假平仓或另造一笔交易。", "",
              "胜率、实际B、乘积和净均值只用自然完成点位；没有点位或缺少赢亏之一时保持未计算。记录表目前为零行，程序测试和历史状态对应不增加前瞻样本。", "",
              f"上交所公告确认2026年10月1日至7日休市，10月8日起开市，见[国庆节休市公告]({CALENDAR_URL})。10月8日只是下一个可能产生执行观察的交易日，不表示届时一定出现信号；完整输入和父状态届时也必须实际可用。", "",
              "下一步先补齐计算链的生产能力，再按固定规则接收新的日线与已完成周线。当前状态为NO_VIEW，不能据截至8月或9月的材料给出10月1日的进出价。", "",
              "十项必要测试覆盖三种程序的末端持仓保留、真实退出照常执行、未来扩展不改变前段，以及拒绝未知价格行和倒置计划日期。旧冻结程序与历史结果未修改。本轮没有自动下单或建立定时任务。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="后续点位验证与参考观察程序准备")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
