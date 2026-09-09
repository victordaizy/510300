"""检查保存的成熟训练、真实持仓状态与预测，交付第106轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.boosted_continuation_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.boosted_continuation_exit_inputs_v1 import boosted_prediction
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

OUT = ROOT / "deliverables/510300非线性继续价值退出_第106轮_20260908"
DOCUMENT = OUT / "非线性继续价值退出_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_BOOSTED_CONTINUATION_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 105 and not OUT.exists(), "106前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时不能直接关闭")
    data = pd.read_parquet(ROOT / cfg["features"])
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    pd.testing.assert_frame_equal(samples, pd.read_parquet(RESEARCH / "original_reference_samples.parquet"))
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    model_checks = []
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in originals], "提升退出月度日程与原模型不同")
    for record, original in zip(models, originals):
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] == original["training_cycles"] and len(rows) == record["training_rows"] == original["training_rows"], "完整成熟训练成员改变")
        require(not len(rows) or rows.exit_index.le(record["fit_index"]).all(), "训练读取未结束参考周期")
        if record["status"] != "FIT_COMPLETE":
            require(record["model"] is None and original["status"] != "FIT_COMPLETE", "无模型来源状态不符")
            continue
        model = record["model"]
        require(model["parameters"] == cfg["model_parameters"] and model["iterations"] == 80 and len(model["trees"]) == 80 and model["features"] == FEATURES, "提升参数或保存树数量改变")
        selected = members[members.fit_index.eq(record["fit_index"])].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_allclose(selected.sample_weight, rows.sample_weight, atol=0, rtol=0)
        np.testing.assert_allclose(selected.groupby("cycle_id").sample_weight.sum(), 1., atol=1e-12, rtol=0)
        require(abs(model["baseline"]-np.average(rows.target, weights=rows.sample_weight)) < 1e-12, "提升初始值不是原周期等权平均")
        require(model["training_check"]["rows"] == len(rows) and model["training_check"]["maximum_library_prediction_difference"] < 1e-12, "保存数值与拟合库预测不符")
        split_nodes, leaves = 0, 0
        for tree in model["trees"]:
            leaf = np.asarray(tree["is_leaf"], bool)
            require(int(leaf.sum()) <= 7 and (np.asarray(tree["sample_count"])[leaf] >= 60).all(), "叶数或每叶最小原始状态数不符")
            require(np.isfinite(tree["value"]).all(), "提升叶值不完整")
            split_nodes += int((~leaf).sum())
            leaves += int(leaf.sum())
        model_checks.append({"fit_origin": record["fit_origin"], "cycles": len(ids), "rows": len(rows), "split_nodes": split_nodes, "leaves": leaves,
                             **model["training_check"]})
    require(len(model_checks) == 114 and result["failed_fits"] == 0, "成熟训练次数或失败状态不符")
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, accounts, predictions, differences = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                owned = ledger[ledger.cycle_id.eq(cycle_id)].set_index("date")
                values = owned.shares*owned.mark+owned.dividend_recognized.cumsum()
                peaks = np.maximum.accumulate(np.r_[cycle.entry_cost_cny, values.to_numpy()])[1:]
                peak = dict(zip(owned.index, peaks))
                count = 0
                for row in group.itertuples():
                    actual = values.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index]]
                    np.testing.assert_allclose([getattr(row, c) for c in FEATURES], expected, atol=1e-12, rtol=0)
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        m = by_date[row.learning_fit_origin]
                        require(m["latest_exit_index"] <= m["fit_index"] <= row.origin_index, "实际提升调用未来模型")
                        value = boosted_prediction(m["model"], expected)
                        require(abs(value-row.continuation_prediction) < 1e-12, "真实持仓的保存提升预测不能复算")
                        count = count+1 if value < 0 else 0
                        predictions.append({"period": period, "cost": cost, "origin": row.origin, "cycle_id": cycle_id, "prediction": value})
                    else:
                        require(pd.isna(row.continuation_prediction), "提升退出无观点时填预测")
                        count = 0
                    require(count == row.negative_confirmation_count and bool(row.learned_exit_requested) == (count >= 2), "提升退出负确认或周期重置不符")
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            actual, m = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(actual[field], m[field]), "提升退出保存指标不符")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            old = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference_vs_original_ridge": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                                "share_quantity_changed_days": int(ledger.shares.ne(old.shares).sum()), "trade_count_difference": actual["trade_count"]-summarize(old, cfg)["trade_count"]})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
    for name, rows in [("saved_model_checks.csv", model_checks), ("saved_prediction_checks.csv", predictions), ("saved_actual_cycles.csv", cycles), ("saved_account_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_BOOSTED_MODELS_MATURE_INPUTS_ACTUAL_STATES_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(model_checks),
               "original_reference_rows_preserved": len(samples), "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
               "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(accounts), "complete_cycles": len(cycles),
               "new_accounts_or_models": 0, "source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第106轮未达标。主基础／压力夏普0.081／0.003，较早0.555／0.533，均弱于原八项线性退出。主基础年化0.33%、最大回撤19.56%；较早年化6.47%、回撤13.79%。模型改变了交易，但没有改善风险收益，关闭这一固定提升退出，不通过树数、学习率、叶数、正则或确认天数救回。"
    status = "COMPLETED_BOOSTED_CONTINUATION_WORSE_BOTH_PERIODS_NOT_TARGET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第106轮之后：隔夜下行风险占比的信息增量", "", decision, "",
        "106六必要测试5.05秒，114拟合及四账户约65.72秒，27原日程无模型、零求解失败、零新参考。主26周期52成交177持股收盘，全有模型；早9周期18成交290收盘，86有模型204无模型。934真实持仓状态、526保存预测、114成熟模型及四账户70周期已核。不要重训或重跑。", "",
        "同一八因子加提升非线性没有改善；前104混合也未改变实际预测方向。下一优先测试不同的信息表达，而非继续增加算法复杂度。研究库存显示第81轮已经做过先半仓再价格确认，修正版使用每日shares恢复正确锚点；本轮只读查重后跳过，不另改成等待天数或换父策略救回。", "",
        "回查第7轮全部期限逆回购，当前定义是新披露公开信息总量；月报实际操作日、预告与到期证据仍不同，不能当作真实每日净投放。现有信息不足以无新增来源地构造严格的未到期总余额，暂不回到慢源补齐。", "",
        "下一107拟定一个新增信息因子：过去二十交易日负隔夜对数收益的平方合计，除以同窗全部隔夜及日内对数收益平方合计。它区分总波动中发生于隔夜下跌的部分；不预设正负系数，不直接硬加清仓阈值。全部二十行必须完整，分母为零或缺失保持未知；已知没有隔夜下跌为零，不能把缺失填零。因子在当日收盘已经可知，不借用未来开盘。", "",
        "限定查询research/docs的隔夜下行半方差或负收益平方相关定义未发现相同继续价值因子实现；既有日内隔夜累计收益、整体下行波动和第65轮短长波动路由早已研究，旧结论保持。19:09:09只读可用性已经完成：reports/research/510300_overnight_downside_share_availability_20260908/result.json。3456行情中3436行有该因子，原1461个D60参考状态全部完整；没有新模型、预测、校准或账户，不重复做可用性。", "",
        "下一拟沿用原八项Ridge1、原周期等权标准化及clip5，追加这一第九项信息；最近20完整成熟周期、min10周期100状态、同141月首114拟合，原进入、6/8/60退出、连续两次负预测、重现与两日冷却全不变。新增信息用于各实际持仓收盘的同一日期，不用未来参考周期状态。原模型较早首成熟日仍2017-03-01，不降低门槛或借新模型救早期。一个固定设置、零新参考，只做定义分红/缺失/时钟及新增输入一致性必要测试，然后冻结训练和四账户。", "",
        "目前107尚未实现或登记。短中文规则和保存数据即可，不长篇展开逐月模型，不做GPT包，不补EPS/公募慢源；完整成本、两段历史与权限不变，目标仍未实现。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第106轮：非线性继续价值退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "六项必要测试通过，114个成熟模型和四个完整账户约65.72秒。原样本、完整周期权重、真实持仓八因子、保存树预测及费用分红净值已核。主历史持股收盘由原252日降到177日，成交由48次增至52次，同时最大回撤扩大；此次非线性退出没有实现有效风险控制。较早历史夏普也从原0.748降至0.555。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 106, "study": result["study_id"], "title": "原八项真实持仓状态的非线性提升退出", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND106_BOOSTED_CONTINUATION_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="106轮，382不同设置，398已评价来源版本，403登记含5旧未运行，1498主评价记录。",
                 next_work={"status": "OVERNIGHT_DOWNSIDE_SHARE_INPUTS_AVAILABLE_NOT_REGISTERED", "focus": "现有二十日隔夜下行风险占比追加到原八项退出；可用性已完成", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="106六测试、114成熟拟合、四新账户及保存核对交付完成；107输入可用，尚未登记。")
    index["deliveries"].append({"created_at": now(), "type": "BOOSTED_CONTINUATION_ROUND106_FAST_CHINESE_RESULTS", "rounds": [106], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相比原线性退出": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
