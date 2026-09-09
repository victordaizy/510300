"""只整理已经保存的进入事件与当时因子，检查样本覆盖，不拟合或新回测。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/research/510300_entry_path_coverage_v1"
OUT = ROOT / "reports/research/510300_entry_context_coverage_20260908"


def main():
    require(not OUT.exists(), "进入事件覆盖已整理，不重复读取生成")
    paths = pd.read_csv(OLD / "reference_paths.csv", parse_dates=["entry_origin", "entry_date", "exit_date"])
    groups = pd.read_csv(OLD / "reference_episodes.csv", parse_dates=["start_date", "end_date", "closed_date", "group_mature_date"])
    data = pd.read_parquet(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet")
    factor = pd.read_parquet(OLD / "entry_factors.parquet")
    require(pd.DatetimeIndex(factor.date).equals(pd.DatetimeIndex(data.date)), "原进入因子日历不符")
    origin = paths.entry_origin_index.to_numpy(int)
    require(pd.DatetimeIndex(paths.entry_origin).equals(pd.DatetimeIndex(data.date.iloc[origin])), "保存进入事件的原点不符")
    fields = ["mom5", "mom20", "sma120", "vol20", "vol_ratio", "z20", "rsi2", "close_location"]
    entries = paths[["path_id", "episode_id", "entry_origin_index", "entry_origin", "status", "entry_index", "exit_index", "resolution_index", "natural_exit"]].copy()
    for field in fields:
        entries[field] = data[field].to_numpy()[origin]
    entries["d60_factor"] = factor.d60_factor.to_numpy()[origin]
    entries["origin_features_complete"] = np.isfinite(entries[[*fields, "d60_factor"]]).all(axis=1)
    entries = entries.merge(groups[["episode_id", "group_mature_index", "group_mature_date", "maturity_status", "reference_left_truncated"]], on="episode_id", how="left", validate="many_to_one")
    snapshots = []
    for date in ["2014-12-31", "2019-12-31", "2026-08-14"]:
        t = int(pd.DatetimeIndex(data.date).searchsorted(pd.Timestamp(date), side="right")-1)
        available = groups[groups.group_mature_index.notna() & groups.group_mature_index.le(t)]
        local = entries[entries.episode_id.isin(available.episode_id)]
        snapshots.append({"as_of": date, "mature_signal_groups": len(available), "paths_in_mature_groups": len(local),
            "naturally_completed_paths_in_mature_groups": int(local.natural_exit.sum()), "complete_entry_context_rows": int(local.origin_features_complete.sum()),
            "overlapping_paths_are_independent_samples": False})
    mature_groups = groups[groups.group_mature_date.notna()].sort_values(["group_mature_date", "episode_id"])
    OUT.mkdir(parents=True)
    entries.to_parquet(OUT / "saved_entry_context_rows.parquet", index=False)
    entries.to_csv(OUT / "原进入事件及当时因子.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(snapshots).to_csv(OUT / "各时点成熟覆盖.csv", index=False, encoding="utf-8-sig")
    result = {"checked_at": now(), "status": "SAVED_ENTRY_CONTEXT_COVERAGE_COMPLETE_NO_NEW_MODEL_OR_ACCOUNT", "saved_entry_paths": len(paths), "saved_signal_groups": len(groups),
        "natural_paths": int(paths.natural_exit.sum()), "unresolved_paths": int(paths.resolution_index.isna().sum()),
        "complete_origin_contexts": int(entries.origin_features_complete.sum()), "mature_signal_groups": len(mature_groups),
        "tenth_group_maturity": str(mature_groups.group_mature_date.iloc[9].date()) if len(mature_groups) >= 10 else None,
        "snapshots": snapshots, "new_strategy_configurations": 0, "new_model_fits": 0, "new_accounts": 0,
        "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [OLD / "reference_paths.csv", OLD / "reference_episodes.csv", OLD / "entry_factors.parquet", ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"]],
        "boundary": "第59轮已检验扩大路径训练继续持有退出；第79轮检验继续持有正负分类。进入前是否接受机会属于不同动作，但当前只整理覆盖，尚未设计或证明新方法。1213重叠路径不是1213个独立市场样本。"}
    write_json(OUT / "result.json", result, exclusive=True)
    note = ROOT / "docs/510300_AFTER_OWN_CUSHION_RISK_20260908.md"
    with note.open("a", encoding="utf-8") as stream:
        stream.write("\n## 第95轮登记前已有输入整理\n\n已直接复用第59轮1213条保存进入路径及49个原信号组，整理各进入收盘的九项市场因子，不读取未来因子，不重跑路径、训练或账户。旧59学习的是进入之后的继续持有价值；旧79也是持仓后的方向分类，两者都不是进入前接受机会的分类。\n\n")
        for snapshot in snapshots:
            stream.write(f"{snapshot['as_of']}已经成熟{snapshot['mature_signal_groups']}个原信号组，覆盖{snapshot['paths_in_mature_groups']}条重叠路径，其中自然完成{snapshot['naturally_completed_paths_in_mature_groups']}条；不能把路径数当作独立样本数。\n\n")
        stream.write(f"第十个完整信号组成熟日期为{result['tenth_group_maturity']}。输入表在reports/research/510300_entry_context_coverage_20260908/saved_entry_context_rows.parquet，完整回执同目录result.json。没有新增训练目标、策略参数或账户；下次直接使用这些保存输入，不重复整理。\n")
    p = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(p.read_text(encoding="utf-8"))
    index.update(updated_at=now(), next_work={"status": "SAVED_ENTRY_CONTEXT_COVERAGE_COMPLETE_METHOD_NOT_REGISTERED", "focus": "区别旧持仓退出学习，利用已有进入前因子和成熟信号组研究接受进入机会", "source": str(note.relative_to(ROOT)), "prepared_inputs": str((OUT / "result.json").relative_to(ROOT))}, process_state_note="94完成；95登记前进入上下文已整理，1213路径49组，零新增模型或账户，尚未登记正式新方法。")
    write_json(p, index)
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
