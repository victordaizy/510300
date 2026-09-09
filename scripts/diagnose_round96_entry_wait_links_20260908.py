"""只检查保存进入事件的等待连接及未成交动作，不训练等待模型或新账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require, now, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
P59 = ROOT / "reports/research/510300_entry_path_coverage_v1"
P95 = ROOT / "reports/research/510300_entry_payoff_gate_v1"
OUT = ROOT / "reports/research/510300_entry_wait_transition_20260908"


def main():
    require(not OUT.exists(), "进入等待连接已检查，不重复生成")
    paths = pd.read_csv(P59 / "reference_paths.csv", parse_dates=["entry_origin", "planned_entry_date"])
    groups = pd.read_csv(P59 / "reference_episodes.csv", parse_dates=["group_mature_date"])
    contexts = pd.read_parquet(ROOT / "reports/research/510300_entry_context_coverage_20260908/saved_entry_context_rows.parquet")
    labels = pd.read_parquet(P95 / "entry_payoff_samples.parquet").set_index("path_id")
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    cfg = json.loads((ROOT / "config/510300_entry_payoff_gate_v1.json").read_text(encoding="utf-8"))
    path_by_origin = {int(row.entry_origin_index): row for row in paths.itertuples()}
    group_by_id = {int(row.episode_id): row for row in groups.itertuples()}
    rows, unfilled_checks = [], []
    for path in paths.itertuples():
        t, group = int(path.entry_origin_index), group_by_id[int(path.episode_id)]
        require(int(path.path_id) == t and factors.date.iloc[t] == path.entry_origin, "保存进入原点编号不符")
        following = path_by_origin.get(t+1)
        next_same = following is not None and int(following.episode_id) == int(path.episode_id)
        terminal = pd.notna(group.closed_index) and int(group.closed_index) == t+1
        if next_same:
            next_id, wait_status, wait_known_date = int(following.path_id), "WAIT_TO_NEXT_CLOSE_IN_SAME_GROUP", factors.date.iloc[t+1]
        elif terminal:
            require(factors.signal_available.iloc[t+1] and factors.raw_entry.iloc[t+1] == 0, "没有确认原条件消失，不能把等待标为终止")
            next_id, wait_status, wait_known_date = None, "WAIT_EPISODE_END_CASH", factors.date.iloc[t+1]
        else:
            next_id, wait_status, wait_known_date = None, "NO_VIEW_UNRESOLVED_WAIT_BOUNDARY", pd.NaT
        payoff, buy_kind, buy_known_date = None, "NO_VIEW_BUY_PATH_NOT_NATURALLY_RESOLVED", pd.NaT
        if path.natural_exit:
            label = labels.loc[int(path.path_id)]
            payoff, buy_kind, buy_known_date = float(label.target), "FILLED_BUY_NATURAL_EXIT_TERMINAL_PAYOFF", label.economic_maturity_date
        elif pd.notna(path.resolution_index) and pd.isna(path.entry_index):
            actual = pd.read_parquet(P59 / "reference_path_ledgers.parquet", filters=[("path_id", "==", int(path.path_id))])
            require(len(actual) == 1 and actual.filled_quantity.eq(0).all() and actual.shares.eq(0).all(), "首次未成交路径的真实状态不符")
            require(actual.equity.eq(cfg["initial_capital"]).all() and actual.commission.eq(0).all() and actual.slippage_cost.eq(0).all(), "未成交后现金并非原本金，不能视作零单步收益")
            require(int(path.resolution_index) == t+1 and actual.date.iloc[0] == factors.date.iloc[t+1], "首次未成交的观察时钟不符")
            payoff, buy_kind, buy_known_date = 0., "UNFILLED_BUY_ZERO_STEP_REWARD_THEN_SAME_CASH_STATE", factors.date.iloc[t+1]
            unfilled_checks.append({"path_id": int(path.path_id), "entry_origin": path.entry_origin, "next_date": buy_known_date, "execution_status": actual.status.iloc[0], "next_path_id": next_id, "continuation_is_required_if_group_remains": next_same})
        mature_date = group.group_mature_date
        eligible = pd.notna(mature_date)
        if eligible:
            require(buy_kind != "NO_VIEW_BUY_PATH_NOT_NATURALLY_RESOLVED" and wait_status != "NO_VIEW_UNRESOLVED_WAIT_BOUNDARY", "成熟原信号组仍缺少买入或等待动作资料")
            require(buy_known_date <= mature_date and wait_known_date <= mature_date, "动作结果可用日超过原整组成熟日")
        rows.append({"path_id": int(path.path_id), "episode_id": int(path.episode_id), "entry_origin": path.entry_origin,
            "signal_age_observed_days": t-int(group.start_index)+1, "buy_outcome_kind": buy_kind,
            "buy_terminal_or_immediate_return": payoff, "buy_outcome_known_date": buy_known_date,
            "wait_outcome_kind": wait_status, "next_path_id": next_id, "next_close_observed_date": wait_known_date,
            "original_group_mature_date": mature_date, "whole_group_available": eligible,
            "next_state_fields_are_future_training_objects_only": True})
    links = pd.DataFrame(rows).merge(contexts.drop(columns=["episode_id", "entry_origin"]), on="path_id", how="left", validate="one_to_one")
    links["next_path_id"] = links.next_path_id.astype("Int64")
    require(links.path_id.is_unique and links.signal_age_observed_days.gt(0).all(), "路径编号或原点可知的信号年龄不符")
    for row in links[links.next_path_id.notna()].itertuples():
        require(int(row.next_path_id) == row.path_id+1, "等待连接跳过交易日")
    OUT.mkdir(parents=True)
    links.to_parquet(OUT / "entry_action_links.parquet", index=False)
    links.to_csv(OUT / "买入与等待的原始动作连接.csv", index=False, encoding="utf-8-sig")
    mature = links[links.whole_group_available]
    result = {"checked_at": now(), "status": "SAVED_ENTRY_WAIT_LINKS_COMPLETE_NO_POLICY_OR_MODEL_REGISTERED", "all_entry_states": len(links), "complete_group_states": len(mature), "complete_groups": int(mature.episode_id.nunique()),
        "all_wait_link_states": int(links.next_path_id.notna().sum()), "complete_group_wait_links": int(mature.next_path_id.notna().sum()),
        "complete_group_wait_ends": int(mature.wait_outcome_kind.eq("WAIT_EPISODE_END_CASH").sum()), "unknown_wait_boundaries": int(links.wait_outcome_kind.eq("NO_VIEW_UNRESOLVED_WAIT_BOUNDARY").sum()),
        "buy_kinds": links.buy_outcome_kind.value_counts().to_dict(), "unfilled_action_checks": unfilled_checks,
        "mature_group_maximum_observed_signal_age": int(mature.signal_age_observed_days.max()),
        "new_strategy_configurations": 0, "new_models": 0, "new_accounts": 0,
        "economic_boundary": "完整买入收益与等待零现金单步收益均以原20万元为单位。首次未成交只说明本步零收益；信号组继续时还须保留下一收盘选择价值，不可将其全部未来价值强填零。组内不同进入交易的退出日期不同，等待决策的目标与时间代价仍须正式定义后才能训练。",
        "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [Path(__file__), P59 / "reference_paths.csv", P59 / "reference_episodes.csv", P59 / "entry_factors.parquet", P95 / "entry_payoff_samples.parquet"]]}
    write_json(OUT / "result.json", result, exclusive=True)
    note = ROOT / "docs/510300_AFTER_ENTRY_PAYOFF_GATE_20260908.md"
    with note.open("a", encoding="utf-8") as stream:
        stream.write("\n## 第96轮登记前动作连接已检查\n\n")
        stream.write(f"原1213个进入状态中，{len(mature)}个属于{mature.episode_id.nunique()}个完整成熟组；成熟组内{result['complete_group_wait_links']}个等待动作连接同组下一交易日，{result['complete_group_wait_ends']}个等待后明确观察到原信号结束。缺失边界不跳日、不接到另一组。完整输入保存为reports/research/510300_entry_wait_transition_20260908/entry_action_links.parquet。\n\n")
        stream.write("唯一首次买入未成交的原路径已经检查实际账本：无成交、无佣金滑点、仍为原现金本金。它的单步收益是已观察的零，但后续若同组仍有机会，现金状态仍有继续选择价值；不能简单当作完整交易零收益标签。输入新增信号已经持续的交易日数量，只从原信号开始日到当前原点计数，不使用组最终长度。\n\n")
        stream.write("第96轮仍没有冻结模型或策略，没有新增账户。下一步直接使用这些连接，先说明等待的终止、不同退出日期的收益比较及训练近似，再登记；不重新抓数据或重做本检查。\n")
    p = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(p.read_text(encoding="utf-8"))
    index.update(updated_at=now(), next_work={"status": "ENTRY_WAIT_ACTION_LINKS_READY_METHOD_NOT_REGISTERED", "focus": "正式定义进入与等待的动作价值和训练近似，复用已检查完整连接", "source": str(note.relative_to(ROOT)), "prepared_inputs": str((OUT / "result.json").relative_to(ROOT))}, process_state_note="95已完成并关闭固定符号筛选；96原买入与等待连接已整理，仍无96新模型或策略账户。")
    write_json(p, index)
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
