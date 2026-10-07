"""本地续算和原冻结交付流程的独立宿主；不持有已失效的旧会话句柄。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finish_factor96_financial_scope_v2_when_ready as pipeline
from research import factor96_financial_parser_resume_v2_01 as recovery


def main():
    pipeline.CONTROL = ROOT / "data/audit/factor96_financial_scope_v2_controller_03"
    pipeline.CONTROL.mkdir(parents=True, exist_ok=True)
    assert not (pipeline.CONTROL / "pipeline_started.json").exists()
    assert recovery.digest(Path(__file__)) == recovery.digest(recovery.RECOVERY / Path(__file__).name)
    assert recovery.digest(ROOT / "scripts/finish_factor96_financial_scope_v2_when_ready.py") == recovery.digest(recovery.RECOVERY / "finish_factor96_financial_scope_v2_when_ready.py")
    pipeline.check_prepared_code()
    pipeline.save("pipeline_started.json", {"at": pipeline.now(), "pid": os.getpid(),
        "status": "LOCAL_PARSE_CONTINUATION_THEN_FROZEN_DELIVERY", "prior_controller_pid": 18352,
        "new_network_requests": 0, "new_accounts": 0, "goal_achieved": False})
    stage = "resume"
    try:
        pipeline.execute_stage(stage, ["-m", "research.factor96_financial_parser_resume_v2_01", "run"])
        assert (pipeline.OUT / "batch_complete.json").exists()
        pipeline.check_prepared_code()
        stage = "replay"
        pipeline.execute_stage(stage, ["-m", "research.factor96_financial_parser_replay_v2", "replay"])
        assert pipeline.read(pipeline.OUT / "source_repair_result.json")["all_confirmed_regressions_passed"]
        assert not (pipeline.OUT / "parser_semantic_invalidation_addendum.json").exists()
        stage = "report"
        pipeline.execute_stage(stage, ["scripts/report_factor96_financial_parser_scope_v2.py"])
        stage = "package"
        pipeline.execute_stage(stage, ["scripts/package_factor96_financial_parser_scope_v2.py"])
        receipt = pipeline.read(pipeline.OUT / "delivery_receipt.json")
        assert receipt["saved_output_recomputation"]["status"] == "PASS_SAVED_V2_SCOPE_REPAIR_AND_UNCHANGED_MEASUREMENT"
        pipeline.save("pipeline_complete.json", {"at": pipeline.now(), "status": "SOURCE_REPLAY_AND_LOCAL_DELIVERY_COMPLETE",
            "zip_path": receipt["zip_path"], "zip_sha256": receipt["sha256"], "T11": "NOT_RUN", "new_accounts": 0,
            "new_network_requests": 0, "goal_achieved": False, "external_review": "NOT_PERFORMED"})
        print("本地续算、同口径重算及交付已完成；T11账户仍未运行。", flush=True)
    except Exception as error:
        pipeline.save("pipeline_failed.json", {"at": pipeline.now(), "stage": stage, "error_type": type(error).__name__,
            "error": str(error), "status": "FAILED_PRESERVE_ARTIFACTS_NO_SOURCE_RETRY", "new_accounts": 0, "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
