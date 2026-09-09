"""生成独立第72轮入口，复用既有完整账户框架，不运行历史。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "research/deleted_cycle_stability_v1.py"
assert not target.exists(), "新入口已经存在，停止覆盖"
source = (ROOT / "research/median_continuation_v1.py").read_text(encoding="utf-8")
source = source.replace("from sklearn.exceptions import ConvergenceWarning\n", "")
source = source.replace("from research.learned_cycle_exit_v1 import training_rows", "from research.learned_cycle_exit_v1 import training_rows, chinese_formula, CN")
source = source.replace("from research.median_continuation_inputs_v1 import fit_median, median_chinese_formula, MedianExitController", "from research.deleted_cycle_stability_inputs_v1 import fit_deletions, DeletedCycleExitController")
source = source.replace("median_continuation_v1", "deleted_cycle_stability_v1").replace("median_continuation_inputs_v1", "deleted_cycle_stability_inputs_v1")
source = source.replace("MEDIAN_CONTINUATION", "DELETED_CYCLE_STABILITY").replace("MedianExitController", "DeletedCycleExitController")
source = source.replace("条件中位数继续价值退出", "逐周期删除一致退出").replace("中位数", "逐周期删除一致")
source = source.replace('"feature_names", "feature_clip"]', '"feature_names", "feature_clip", "ridge_alpha"]')
source = source.replace('study_id="510300_DELETED_CYCLE_STABILITY_V1", round=71', 'study_id="510300_DELETED_CYCLE_EXIT_STABILITY_V1", round=72')
source = source.replace('quantile=.5, alpha=0., fit_intercept=True, solver="highs", model_loss="CYCLE_EQUAL_WEIGHTED_ABSOLUTE_ERROR",',
    'fit_intercept=True, solver="svd", model_loss="CYCLE_EQUAL_WEIGHTED_SQUARE_ERROR_WITH_ORIGINAL_RIDGE_PENALTY", planned_deletion_fits=2055, all_members_must_be_negative=True, deletion_backfill=False,')
source = source.replace('docs/510300_DELETED_CYCLE_STABILITY_V1.md', 'docs/510300_DELETED_CYCLE_EXIT_STABILITY_V1.md')
source = source.replace('previous_goal_turn_classification="PROGRESS_ROUND70_COMPLETED_AND_DELIVERED"', 'previous_goal_turn_classification="PROGRESS_ROUND71_COMPLETED_AND_DELIVERED"')
source = source.replace('ROOT / "tests/test_deleted_cycle_stability_v1.py", OUT / "tests_receipt.json"',
    'ROOT / "tests/test_deleted_cycle_stability_v1.py", ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json"')
source = source.replace("第71轮", "第72轮")
start, end = source.index("def train_models("), source.index("def run():")
train = '''def train_models(data, samples, originals, cfg):
    models, receipts, memberships, deletion_receipts, coefficient_rows = [], [], [], [], []
    chinese = ["# 每月逐周期删除退出模型：全部因子系数与中文规则", "", "原模型不重拟合；每个已成熟主样本的全部删除模型均保存。表中小数采用10位展示，实际计算使用保存完整精度。", ""]
    attempts = 0
    for position, original in enumerate(originals):
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "原模型与本轮成熟样本不同")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (original["status"] == "FIT_COMPLETE"), "成熟模型支持时点改变")
        stored, failure, deletions = None, None, []
        status = "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"
        if eligible:
            deletions = fit_deletions(rows, t, cfg)
            attempts += len(deletions)
            require(len(deletions) == len(ids), "没有尝试全部原周期的删除")
            if all(d["status"] == "FIT_COMPLETE" for d in deletions):
                stored = {"kind": "DELETED_CYCLE_COMMITTEE", "base_model": original["model"], "deletions": deletions}
                status = "FIT_COMPLETE"
            else:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", "至少一个删除模型数值求解失败，整组学习判断缺失"
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
            "status": status, "eligible_for_fit": eligible, "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
            "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None, "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
            "failure": failure, "deletion_fits_attempted": len(deletions), "deletion_fits_completed": sum(d["status"] == "FIT_COMPLETE" for d in deletions),
            "model": stored, "failed_month_deletion_details": deletions if failure else None}
        require(not len(rows) or (rows.exit_index <= t).all(), "参考周期在训练时点尚未完成")
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles", "failed_month_deletion_details"]})
        if eligible:
            memberships.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                "sample_weight": float(r.sample_weight), "fit_status": status} for r in rows.itertuples())
        chinese += [f"## {record['fit_origin']}", "", f"主样本已完成周期{len(ids)}个，状态{len(rows)}条，尝试删除模型{len(deletions)}个。", ""]
        formula_models = []
        if eligible:
            formula_models.append(("原全部周期模型，直接复用", original["model"]))
        for deleted in deletions:
            deletion_receipts.append({"fit_index": t, "fit_origin": record["fit_origin"], **{k: v for k, v in deleted.items() if k != "model"}})
            if deleted["model"] is not None:
                formula_models.append((f"删除原周期{deleted['deleted_cycle_id']}后的模型", deleted["model"]))
            else:
                chinese += [f"删除原周期{deleted['deleted_cycle_id']}后求解失败，本月整组学习判断不可用。", ""]
        if not formula_models:
            chinese += ["本月成熟样本不足，没有学习模型，继续原价格及时间退出。", ""]
        for label, model in formula_models:
            chinese += [f"### {label}", ""]
            formula = chinese_formula(model)
            chinese += [formula[0], "", *formula[1:], ""]
            coefficient_rows.extend({"模型训练日": record["fit_origin"], "模型说明": label, "因子": name, "训练均值": mean,
                "训练标准差": scale, "标准化系数": coef, "截距": model["intercept"], "截断绝对值": model["feature_clip"]}
                for name, mean, scale, coef in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
        if (position + 1) % 20 == 0:
            print(f"已处理{position + 1}个原月度时点，累计完成{attempts}次删除拟合尝试。", flush=True)
    require(attempts == cfg["planned_deletion_fits"], "实际删除拟合次数与登记不符")
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": cfg["feature_names"], "loss": cfg["model_loss"]}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(deletion_receipts).to_csv(OUT / "deletion_training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(OUT / "全部月度模型系数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月删除周期退出模型中文规则.md").write_text("\\n".join(chinese) + "\\n", encoding="utf-8")
    print(f"原成熟时点不变：完整模型组{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次，共{attempts}个删除模型。", flush=True)
    return models, receipts


'''
source = source[:start] + train + source[end:]
source = source.replace('"new_model_fits": sum(r["eligible_for_fit"] for r in receipts), "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts),',
    '"new_model_fits": sum(r["deletion_fits_attempted"] for r in receipts), "completed_fits": sum(r["deletion_fits_completed"] for r in receipts),\n        "complete_model_updates": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "failed_submodel_fits": sum(r["deletion_fits_attempted"] - r["deletion_fits_completed"] for r in receipts),')
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print("第72轮完整入口已生成，尚未登记、拟合或运行账户。")
