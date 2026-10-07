"""仅适配冻结账户的毫秒日期表示，原金融算法和新模型用途不变。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research import macro_technical_first_passage_study_v1 as study
from research.point_first_passage_study_v1 import read, write_json, digest, require, now

OLD = study.OUT
OUT = OLD.with_name(OLD.name + "_clock_adapter")
ORIGINAL_ACCOUNT = study.execution.account


def account(data, dividends, parents, risks, signals, cost, start, mode):
    # 只转换午夜日线的存储单位，转换前后绝对时点必须逐个相等。
    dtype = pd.read_parquet(study.BASE / "results/原点全部技术特征.parquet", columns=["date"]).date.dtype
    d, s = data.copy(), signals.copy()
    for frame in [d, s]:
        before = frame.date.astype("datetime64[ns]")
        converted = frame.date.astype(dtype)
        require(np.array_equal(before.to_numpy(), converted.astype("datetime64[ns]").to_numpy()), "日期转换改变时点，停止账户。")
        frame["date"] = converted
    return ORIGINAL_ACCOUNT(d, dividends, parents, risks, s, cost, start, mode)


def freeze():
    require(not (OUT / "protocol.json").exists(), "日期适配协议已存在，不覆盖。")
    original_protocol = study.verify_sources()
    require((OLD / "RUN_STARTED.json").exists(), "没有已确认的原运行入口。")
    require(not (OLD / "saved_models.json").exists() and not (OLD / "control_preflight.json").exists(), "原实验已产生历史模型或完成控制，不能重跑。")
    OUT.mkdir(parents=True, exist_ok=True)
    failure = {"at": now(), "status": "TERMINAL_CONTROL_FORMAT_FAILURE_BEFORE_MODEL_FIT",
               "exit_code": 1, "error": "首个原A控制date的datetime64[ns]与保存datetime64[ms]不同",
               "original_A_control_attempts_before_failure": 1, "new_historical_model_fits": 0,
               "new_candidate_accounts": 0, "market_or_model_parameters_changed": False}
    write_json(OUT / "prior_terminal_failure.json", failure, exclusive=True)
    shutil.copytree(OLD / "results", OUT / "results")
    for name in ["explanation_summary.json", "tests_receipt.json"]:
        shutil.copy2(OLD / name, OUT / name)
    protocol = dict(original_protocol)
    protocol["original_frozen_protocol"] = (OLD / "protocol.json").relative_to(ROOT).as_posix()
    protocol["date_representation_adapter"] = "仅把账户输入date转成原冻结日线存储单位，逐个时点必须相等；不改变日期、计算值、预测或交易顺序。"
    protocol["original_A_control_attempts_before_adapter"] = 1
    protocol["sources"] = list(protocol["sources"])
    for p in [Path(__file__), OLD / "protocol.json", OLD / "RUN_STARTED.json",
              OLD / "results/全部3488当时已知技术与宏观_未知保留.parquet", OLD / "explanation_summary.json",
              OUT / "prior_terminal_failure.json"]:
        protocol["sources"].append({"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)})
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("日期表示适配已登记；复用解释和八项验证，历史模型及候选账户仍未运行。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="首次边界联合用途的原账户日期表示适配。")
    parser.add_argument("command", choices=("freeze", "run", "deliver"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        study.OUT = OUT
        study.execution = SimpleNamespace(account=account)
        {"run": study.run, "deliver": study.deliver}[args.command]()
