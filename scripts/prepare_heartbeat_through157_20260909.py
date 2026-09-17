"""依据157完成索引更新原持续任务的提示参数并保持原排程。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 157 and not index["goal_achieved"] and not index["running_studies"], "157完成状态不同")
    prompt = (ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH157_20260909.md").read_text(encoding="utf-8").rstrip()
    previous = tomllib.loads(Path("E:/CodexData/.codex/automations/510300-1-2/automation.toml").read_text(encoding="utf-8"))
    require(previous["id"] == "510300-1-2" and previous["kind"] == "heartbeat", "持续任务身份改变")
    args = {"id": previous["id"], "mode": "update", "kind": previous["kind"], "name": previous["name"], "prompt": prompt,
        "rrule": previous["rrule"], "status": previous["status"], "targetThreadId": previous["target_thread_id"]}
    if "notification_policy" in previous:
        args["notificationPolicy"] = previous["notification_policy"]
    target = ROOT / "reports/research/510300_heartbeat_update_through157_args.json"
    write_json(target, args, exclusive=True)
    print(json.dumps({"参数文件": str(target), "提示字数": len(prompt), "状态": previous["status"], "计数": index["count_warning"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
