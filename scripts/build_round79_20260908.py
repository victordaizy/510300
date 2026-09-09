"""复用成熟样本和账户流程，建立新的单一概率退出来源。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "research/directional_continuation_v1.py"
    if target.exists():
        raise ValueError("概率退出来源已经存在，不覆盖")
    source = (ROOT / "research/position_state_exit_v1.py").read_text(encoding="utf-8")
    source = source.replace("position_state", "directional_continuation").replace("POSITION_STATE", "DIRECTIONAL_CONTINUATION")
    source = source.replace("position_chinese_formula", "directional_chinese_formula").replace("PositionStateExitController", "DirectionalExitController")
    source = source.replace("三项持仓", "方向概率").replace("第73轮", "第79轮").replace("round=73", "round=79")
    source = source.replace('"feature_clip", "ridge_alpha"', '"feature_clip"')
    source = source.replace('fit_intercept=True, solver="svd", model_loss="CYCLE_EQUAL_WEIGHTED_SQUARE_ERROR_WITH_ORIGINAL_RIDGE_PENALTY",',
        'fit_intercept=True, solver="lbfgs", logistic_C=1., l1_ratio=0., tolerance=1e-8, maximum_iterations=1000, class_weight=None,\n        model_loss="CYCLE_EQUAL_WEIGHTED_BINARY_LOG_LOSS_L2", positive_class="ORIGINAL_CONTINUATION_TARGET_STRICTLY_POSITIVE", exit_probability=.5,')
    source = source.replace(', dropped_feature_columns=["entry_mode", "mom5", "mom20", "sma120", "vol20"]', '')
    source = source.replace('PROGRESS_ROUND72_COMPLETED_AND_DELIVERED', 'PROGRESS_ROUND78_COMPLETED_AND_DELIVERED')
    source = source.replace('status = "FIT_COMPLETE"\n            except', 'status = "FIT_COMPLETE" if stored is not None else "NO_VIEW_SINGLE_CLASS"\n            except')
    source = source.replace('"new_model_fits": sum(r["eligible_for_fit"] for r in receipts)', '"new_model_fits": sum(r["status"] in {"FIT_COMPLETE", "NO_VIEW_MODEL_FIT_FAILED"} for r in receipts)')
    source = source.replace('"failed_fits": sum(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in receipts),', '"failed_fits": sum(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in receipts),\n        "single_class_no_view": sum(r["status"] == "NO_VIEW_SINGLE_CLASS" for r in receipts),')
    source = source.replace('            save_account(folder, PRIMARY, ledger, decisions)',
        '            for saved_frame, column in [(ledger, "execution_reasons"), (decisions, "exit_reasons"), (cycles, "exit_reasons")]:\n                saved_frame[column] = saved_frame[column].str.replace("连续两个收盘预测继续持有收益为负，学习条件请求退出", "连续两个收盘预测继续占优概率低于一半，学习条件请求退出", regex=False)\n            save_account(folder, PRIMARY, ledger, decisions)')
    source = source.replace('三项输入', '八项原输入').replace('三项实际持仓继续收益退出', '继续占优概率退出')
    source = source.replace('directional_continuation_exit_inputs_v1', 'directional_continuation_inputs_v1')
    source = source.replace('510300_directional_continuation_exit_v1', '510300_directional_continuation_v1')
    source = source.replace('test_directional_continuation_exit_v1', 'test_directional_continuation_v1')
    target.write_text(source, encoding="utf-8")
    print("第79轮新概率模型运行来源已生成，旧模型和账户来源保持。")


if __name__ == "__main__":
    main()
