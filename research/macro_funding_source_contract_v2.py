"""仅替换资金源支持语义副本，继承原冻结模型与账户，原失败保持。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd

from research import funding_availability_semantics_v1 as diagnosis
from research import macro_technical_first_passage_clock_adapter_v1 as clock_adapter
from research import macro_technical_first_passage_study_v1 as study

ROOT = study.ROOT
OLD = clock_adapter.OUT
OUT = ROOT / "reports/research/510300_macro_funding_source_contract_v2"
CARD = ROOT / "docs/510300_MACRO_FUNDING_SOURCE_CONTRACT_V2.md"
WRITE = study.write_json


def metadata_write(path, value, exclusive=False):
    if path == OUT / "summary.json":
        value = dict(value)
        value.update(decision="TECH.R198", inherited_engine_decision="TECH.R192",
                     source_correction_decision="TECH.R196", source_semantics_contract="SOURCE_CLOCKS_NOT_OLD_QUANTILE_SIGNAL_SUPPORT",
                     original_rejected_result_preserved=True)
    return WRITE(path, value, exclusive=exclusive)


def configure():
    study.OUT = OUT
    study.FUNDING = OUT / "inputs/funding_source_known.parquet"
    study.execution = SimpleNamespace(account=clock_adapter.account)
    study.write_json = metadata_write


def freeze():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("源合同修正已登记，不覆盖。")
    tests = study.read(OUT / "source_contract_tests_receipt.json")
    study.require(tests["passed"] == 5 and tests["exit_code"] == 0, "五项源合同测试未通过。")
    study.require(tests["source_mask_code_sha256"] == study.digest(Path(diagnosis.__file__)), "测试对应的源合同代码改变。")
    prior = study.read(diagnosis.OUT / "summary.json")
    study.require(prior["recent_training_contracts_exactly_same"] is False, "没有需要另测的近期训练合同变化。")
    protocol = dict(study.read(OLD / "protocol.json"))
    for source in protocol["sources"]:
        study.require(study.digest(ROOT / source["path"]) == source["sha256"], "旧输入源改变：" + source["path"])
    (OUT / "inputs").mkdir(exist_ok=True)
    (OUT / "code").mkdir(exist_ok=True)
    f = pd.read_parquet(diagnosis.FUND)
    fixed = f.copy()
    fixed["legacy_quantile_signal_supported"] = f.fund_known
    fixed["fund_known"] = diagnosis.source_mask(f)
    pd.testing.assert_frame_equal(f.drop(columns="fund_known"), fixed[f.columns].drop(columns="fund_known"), check_exact=True)
    fixed.to_parquet(OUT / "inputs/funding_source_known.parquet", index=False)
    for source in [Path(__file__), Path(diagnosis.__file__), Path(clock_adapter.__file__)]:
        shutil.copy2(source, OUT / "code" / source.name)
    paths = [Path(__file__), Path(diagnosis.__file__), CARD, ROOT / "tests/test_funding_source_contract_v2.py",
             OUT / "source_contract_tests_receipt.json", OUT / "inputs/funding_source_known.parquet",
             diagnosis.OUT / "protocol.json", diagnosis.OUT / "summary.json", OLD / "protocol.json", OLD / "summary.json"]
    protocol.update(at=study.now(), study="510300_MACRO_FUNDING_SOURCE_CONTRACT_V2", decision="TECH.R197",
                    complete_purpose=CARD.relative_to(ROOT).as_posix(), source_mask_only_change=True,
                    additional_joint_origins=prior["additional_joint_origins"], changed_training_contracts=prior["changed_training_contracts"],
                    original_frozen_protocol=(OLD / "protocol.json").relative_to(ROOT).as_posix(),
                    original_A_control_attempts_before_adapter=0, new_source_contract_tests_passed=5,
                    parent_eight_tests_previously_passed_not_rerun=True)
    protocol["sources"] = list(protocol["sources"]) + [{"path": p.absolute().relative_to(ROOT).as_posix(), "sha256": study.digest(p)} for p in paths]
    WRITE(OUT / "protocol.json", protocol, exclusive=True)
    print("TECH.R197唯一源语义修正金融用途已登记；原数值、模型和金融参数不改。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宏观资金源合同修正的隔离完整检验")
    parser.add_argument("action", choices=["freeze", "explain", "run", "deliver"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        configure()
        {"explain": study.explain, "run": study.run, "deliver": study.deliver}[args.action]()
