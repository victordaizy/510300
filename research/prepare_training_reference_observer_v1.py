"""为原退出模型的训练参考派生真实收盘观察程序，保留非再武装的原始进入规则。"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research.prepare_reference_observation_engines_v1 import replace_once

SOURCE = ROOT / "research/learned_cycle_exit_account_v1.py"
DESTINATION = ROOT / "research/training_reference_observation_v1.py"
OUT = ROOT / "reports/research/510300_point_training_observation_v1"


def main():
    if DESTINATION.exists():
        raise FileExistsError("训练参考观察程序已存在，不覆盖。")
    original = SOURCE.read_text(encoding="utf-8-sig")
    function = next(node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == "simulate_learned_exit")
    code = "\n".join(original.splitlines()[function.lineno-1:function.end_lineno])
    code = replace_once(code, "def simulate_learned_exit(", "def observe_training_reference(")
    code = replace_once(code, "spec, controller=None):", "spec, controller=None, *, next_execution_date):")
    needle = "    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])"
    code = replace_once(code, needle, "    next_execution_date = validate_observation_boundary(data, next_execution_date)\n" + needle)
    code = replace_once(code, "dates[t + 1] if t + 1 < len(dates) else pd.NaT", "dates[t + 1] if t + 1 < len(dates) else next_execution_date")
    code = replace_once(code, "terminal = day == last", "terminal = False  # 文件结束不构成训练参考的自然退出。")
    code = replace_once(code, '"研究终点退出未成交"', '"观察截止仍持有，尚无自然退出"')
    imports = '''"""原退出模型训练参考的真实收盘观察版，由固定源程序派生。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require
from research.reference_observation_accounts_v1 import validate_observation_boundary


'''
    generated = imports + code + "\n"
    ast.parse(generated)
    DESTINATION.write_text(generated, encoding="utf-8")
    common.save_json(OUT / "source_derivation.json", {
        "at": common.now(), "source": str(SOURCE.relative_to(ROOT)), "source_sha256": common.digest(SOURCE),
        "source_function": "simulate_learned_exit", "destination": str(DESTINATION.relative_to(ROOT)),
        "destination_sha256": common.digest(DESTINATION), "generator_sha256": common.digest(Path(__file__)),
        "changes": ["最后真实收盘不因文件结束强平", "保留末日决定并接调用方提供的下一执行日", "未自然结束周期保留未完成状态"],
        "original_entry_rules_preserved": True, "rearm_condition_added": False, "original_source_modified": False,
    })
    print("原训练参考的真实收盘观察程序已生成，原进入与退出规则保留。", flush=True)


if __name__ == "__main__":
    main()
