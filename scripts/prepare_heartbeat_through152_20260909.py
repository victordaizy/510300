"""只准备持续任务更新参数，保留原任务字段和通知偏好。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = tomllib.loads(Path("E:/CodexData/.codex/automations/510300-1-2/automation.toml").read_text(encoding="utf-8"))
    require(old["id"] == "510300-1-2" and old["kind"] == "heartbeat", "持续任务身份不同")
    index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 152 and not index["goal_achieved"], "第152轮持续任务状态不同")
    prompt = (ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH152_20260909.md").read_text(encoding="utf-8").rstrip()
    args = {"id": old["id"], "mode": "update", "kind": old["kind"], "name": old["name"], "prompt": prompt,
            "rrule": old["rrule"], "status": old["status"], "targetThreadId": old["target_thread_id"]}
    if "notification_policy" in old:
        args["notificationPolicy"] = old["notification_policy"]
    path = ROOT / "reports/research/510300_heartbeat_update_through152_args.json"
    write_json(path, args, exclusive=True)
    print(json.dumps({"参数文件": str(path), "提示字数": len(prompt), "状态": old["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
