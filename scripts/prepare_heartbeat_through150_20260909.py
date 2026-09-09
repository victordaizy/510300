"""读取现有持续任务字段，生成完整更新参数，不直接修改任务配置。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Path("E:/CodexData/.codex/automations/510300-1-2/automation.toml")
    old = tomllib.loads(source.read_text(encoding="utf-8"))
    require(old["id"] == "510300-1-2" and old["kind"] == "heartbeat", "持续任务身份不同")
    prompt = (ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH150_20260909.md").read_text(encoding="utf-8").rstrip()
    args = {"id": old["id"], "mode": "update", "kind": old["kind"], "name": old["name"],
        "prompt": prompt, "rrule": old["rrule"], "status": old["status"], "targetThreadId": old["target_thread_id"]}
    if "notification_policy" in old:
        args["notificationPolicy"] = old["notification_policy"]
    path = ROOT / "reports/research/510300_heartbeat_update_through150_args.json"
    write_json(path, args, exclusive=True)
    print(json.dumps({"参数文件": str(path), "任务": old["name"], "状态": old["status"], "提示字数": len(prompt)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
