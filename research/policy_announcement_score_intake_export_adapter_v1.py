"""隔离修复政策元数据混合类型的Parquet保存，不改信息时钟或任何金融规则。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from research import policy_announcement_score_intake_v1 as parent

FAILED = parent.OUT
OUT = parent.ROOT / "reports/research/510300_policy_announcement_score_intake_v1_export_adapter"
FORMATS = []


def export(name, frame):
    frame = frame.copy()
    adjusted = []
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            )
            types = {type(value) for value in frame[column].dropna()}
            if str in types and len(types) > 1:
                frame[column] = frame[column].astype("string")
                adjusted.append(column)
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    FORMATS.append({"table": name, "metadata_columns_serialized_as_text": adjusted,
                    "numeric_feature_values_changed": False})


def freeze():
    if not (FAILED / "RUN_STARTED.json").is_file() or (FAILED / "summary.json").exists():
        raise ValueError("只承接已发生、尚无结果的元数据保存错误。")
    receipt = FAILED / "terminal_format_failure.json"
    if not receipt.exists():
        parent.save(receipt, {
            "at": parent.now(), "status": "TERMINAL_ARROW_MIXED_METADATA_AMOUNT_EXPORT_FAILURE",
            "message": "政策amount列含数值与空字符串，PyArrow无法按double保存。24节点原文检查已通过，首张CSV已保存，尚无每日连接、模型或账户。",
            "new_accounts": 0, "new_fits": 0, "new_return_labels": 0,
            "rerun_original": False, "original_code_and_protocol_preserved": True,
        })
    parent.OUT = OUT
    parent.freeze()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    parent.save(OUT / "format_adapter_protocol.json", {
        "at": parent.now(), "failed_output": parent.rel(FAILED),
        "parent_protocol_sha256": parent.digest(FAILED / "protocol.json"),
        "failure_receipt_sha256": parent.digest(receipt),
        "adapter_sha256": parent.digest(Path(__file__)),
        "only_change": "元数据对象列同时含数字与字符串时按文本保存；数值特征、窗口、来源上界、样本、评分与准入合同不改。",
        "new_accounts": 0, "new_fits": 0,
    })
    print("保存格式适配已隔离登记，原失败不覆盖。")


def run():
    cfg = parent.load(OUT / "format_adapter_protocol.json")
    if parent.digest(Path(__file__)) != cfg["adapter_sha256"]:
        raise ValueError("格式适配代码改变。")
    parent.OUT = OUT
    parent.export = export
    parent.run()
    parent.save(OUT / "format_adapter_receipt.json", {
        "at": parent.now(), "status": "PASS_METADATA_SERIALIZATION_ONLY",
        "tables": FORMATS, "new_accounts": 0, "new_fits": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="政策资料保存格式的隔离适配")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
