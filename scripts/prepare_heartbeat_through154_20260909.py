"""读取154已完成索引，生成保持原持续任务字段的更新参数。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 154 and not index["goal_achieved"] and not index["running_studies"], "154完成状态或运行中状态不同")
    prompt = (ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH154_20260909.md").read_text(encoding="utf-8").rstrip()
    old = tomllib.loads(Path("E:/CodexData/.codex/automations/510300-1-2/automation.toml").read_text(encoding="utf-8"))
    require(old["id"] == "510300-1-2" and old["kind"] == "heartbeat", "持续任务身份改变")
    args = {"id": old["id"], "mode": "update", "kind": old["kind"], "name": old["name"], "prompt": prompt,
        "rrule": old["rrule"], "status": old["status"], "targetThreadId": old["target_thread_id"]}
    if "notification_policy" in old:
        args["notificationPolicy"] = old["notification_policy"]
    path = ROOT / "reports/research/510300_heartbeat_update_through154_args.json"
    write_json(path, args, exclusive=True)
    print(json.dumps({"参数文件": str(path), "提示字数": len(prompt), "状态": old["status"], "计数": index["count_warning"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
