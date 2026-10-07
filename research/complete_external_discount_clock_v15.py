"""保存本轮历史诊断的完成证据，保持总研究目标未达成。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_external_discount_clock_v15"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(name):
    return sha256((OUT / name).read_bytes()).hexdigest()


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮完成记录已经存在，不覆盖。")
    verify = load("verification.json")
    assert verify["status"] == "PASS_SAVED_SOURCE_FACT_CLOCK_AND_PATH_RECOMPUTATION"
    frozen = load("freeze.json")
    assert digest("protocol.json") == frozen["protocol_sha256"]
    for row in frozen["inputs"]:
        assert digest("inputs/" + row["name"]) == row["sha256"]
    result = load("result.json")
    assert not result["goal_achieved"] and not result["independent_validation"]
    result["status"] = "COMPLETED_HISTORICAL_MECHANISM_DIAGNOSTIC"
    result["continuation_classification"] = "PROGRESS"
    save("result.json", result)
    stamp = datetime.now().astimezone().isoformat()
    progress = {"at": stamp, "classification": "PROGRESS", "goal_achieved": False,
                "completed": "两个既定八月病例的外部利率、人民币中间价、13份官方公告及原40日路径对齐；63个数值从公告文本复核。",
                "finding": "国内资金松与外部利率升可以同时发生；起点信息和以后公告需分开，消息表不能直接变成确定性的收益判断。",
                "remaining_gap": "未识别对510300各成分股现金流、估值与风险补偿的独立传导贡献，未得到独立验证的方向优势。",
                "next_historical_question": "在相同原窗口内，先查当时已知的行业及企业外部暴露与收益贡献，保留国内事件和盈利变化等竞争解释。",
                "new_models": 0, "new_accounts": 0, "orders_authorized": False,
                "authority": "只做历史观察；不改变其他任务维护的全局配置，不恢复冻结失败候选。"}
    save("goal_progress.json", progress)
    figure = "figures/外部利率汇率与原二十日路径.png"
    report = "第十五轮_信贷改善与外部利率消息时序.md"
    receipt = {"at": stamp, "status": verify["status"], "continuation_classification": "PROGRESS",
               "goal_achieved": False, "independent_validation": False, "checks": verify["checks"],
               "frozen_inputs_and_protocol_unchanged": True,
               "visual_check": {"status": "PASS_VISUAL_INSPECTION", "figure_sha256": digest(figure),
                                "notes": "已查看最终导出图；完整路径、中文标签及消息日期可辨，数值未超出图轴，端点标注已避让曲线。"},
               "report_sha256": digest(report), "result_sha256": digest("result.json"),
               "verification_sha256": digest("verification.json"),
               "scope_limit": "复算是原件值、时钟与会计恒等式检查，不是外部独立审阅、因果识别或成本后收益优势。"}
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    save("completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
