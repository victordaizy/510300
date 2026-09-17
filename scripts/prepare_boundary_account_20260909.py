"""静态复用原完整账户和批运行流程，仅开放独立的请求函数与账户入口。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/boundary_rebalance_account_v1.py"
    batch_destination = ROOT / "research/saved_target_custom_account_runner_v1.py"
    require(not destination.exists() and not batch_destination.exists(), "边界账户或自定义批入口已经存在")
    source = (ROOT / "research/event_clock_account_v1.py").read_text(encoding="utf-8")
    source = source.replace("按事件时钟调整的完整账户，沿用既有费用、整手、分红和T+1记账。", "使用独立边界请求的完整账户，保留原费用、整手、分红和T+1记账。", 1)
    source = source.replace(", target_request\n", "\nfrom research.boundary_rebalance_inputs_v1 import boundary_request as target_request\n", 1)
    source = source.replace("def simulate_event_account(", "def simulate_boundary_account(", 1)
    marker = "    require(event_mask is not None"
    require(source.count(marker) == 1 and "boundary_request as target_request" in source, "边界账户静态替换位置不同")
    source = source.replace(marker, '    require(model_id == "BOUNDARY_REBALANCE" and prediction is None and targets is not None, "本入口仅支持边界目标账户")\n'+marker, 1)
    destination.write_text(source, encoding="utf-8")
    batch = (ROOT / "research/saved_target_batch_runner_v1.py").read_text(encoding="utf-8")
    batch = batch.replace("共读数据和保存对照，一次运行事先列明的有限目标候选。", "共读数据与保存对照，使用显式传入的完整账户函数运行有限候选。", 1)
    batch = batch.replace("from research.event_clock_account_v1 import simulate_event_account\n", "", 1)
    batch = batch.replace("def run_saved_target_batch(root, out, config_path, candidates, parents, controls, build_frames):",
                          "def run_saved_target_custom_account(root, out, config_path, candidates, parents, controls, build_frames, simulate_account):", 1)
    batch = batch.replace("ledger, decisions = simulate_event_account(", "ledger, decisions = simulate_account(", 1)
    require("simulate_event_account" not in batch and "def run_saved_target_custom_account" in batch, "自定义批入口替换不完整")
    batch_destination.write_text(batch, encoding="utf-8")
    print("独立边界账户及可复用批入口已生成，原冻结文件保持。", flush=True)


if __name__ == "__main__":
    main()
