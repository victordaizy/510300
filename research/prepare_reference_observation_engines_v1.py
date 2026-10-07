"""从已固定参考程序派生完整收盘观察版，保留所有经济规则和旧源文件。"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common

DESTINATION = ROOT / "research/reference_observation_accounts_v1.py"
OUT = ROOT / "reports/research/510300_point_forward_readiness_v1"
SOURCES = [
    ("rearmed_cycle_exit_account_v1.py", "simulate_rearmed_exit", "observe_rearmed_exit"),
    ("event_clock_account_v1.py", "simulate_event_account", "observe_event_account"),
    ("simple_price_entry_exit_v1.py", "simulate_policy", "observe_price_policy"),
]


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("原程序边界结构不唯一，停止派生：" + before)
    return text.replace(before, after, 1)


def main():
    if DESTINATION.exists():
        raise FileExistsError("收盘观察版已存在，不覆盖。")
    fragments = ['''"""真实收盘参考状态：保留原交易规则，观察末端不强平、不虚构未来行情。

由 prepare_reference_observation_engines_v1.py 从已存参考函数派生。
来源和边界修改在本轮 source_derivation.json 中逐项记录。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, fill_price, choose_order, execute_order, require, target_request


def validate_observation_boundary(data, next_execution_date):
    dates = pd.DatetimeIndex(data.date)
    require(len(dates) >= 2 and dates.is_unique and dates.is_monotonic_increasing, "观察资料必须为完整递增交易日。")
    require(np.isfinite(data[["open", "close"]].to_numpy(float)).all() and data[["open", "close"]].gt(0).all().all(), "不能以未知或伪造未来价格补观察末端。")
    planned = pd.Timestamp(next_execution_date)
    require(planned == planned.normalize() and planned > dates[-1], "计划执行日必须晚于已知收盘日。")
    return planned
''']
    sources = []
    for filename, original_name, new_name in SOURCES:
        path = ROOT / "research" / filename
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == original_name)
        code = "\n".join(text.splitlines()[function.lineno-1:function.end_lineno])
        code = replace_once(code, "def " + original_name + "(", "def " + new_name + "(")
        if original_name == "simulate_event_account":
            code = replace_once(code, "event_mask: np.ndarray | None = None) ->", "event_mask: np.ndarray | None = None, *, next_execution_date) ->")
        elif original_name == "simulate_rearmed_exit":
            code = replace_once(code, "spec, controller=None):", "spec, controller=None, *, next_execution_date):")
        else:
            code = replace_once(code, "rule, spec):", "rule, spec, *, next_execution_date):")
        needle = "    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])"
        code = replace_once(code, needle, "    next_execution_date = validate_observation_boundary(data, next_execution_date)\n" + needle)
        code = replace_once(code, "dates[t + 1] if t + 1 < len(dates) else pd.NaT", "dates[t + 1] if t + 1 < len(dates) else next_execution_date")
        code = replace_once(code, "terminal = day == last", "terminal = False  # 每行是真实交易日，文件结束不构成卖出理由。")
        if original_name != "simulate_event_account":
            code = replace_once(code, '"研究终点退出未成交"', '"观察截止仍持有，尚无自然退出"')
        fragments.append(code)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": common.digest(path), "source_function": original_name,
                        "generated_function": new_name, "source_first_line": function.lineno, "source_last_line": function.end_lineno})
    generated = "\n\n\n".join(fragments) + "\n"
    ast.parse(generated)
    DESTINATION.write_text(generated, encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    common.save_json(OUT / "source_derivation.json", {
        "at": common.now(), "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": common.digest(Path(__file__)),
        "sources": sources, "destination": str(DESTINATION.relative_to(ROOT)), "destination_sha256": common.digest(DESTINATION),
        "changes": ["计划下一执行日期由调用方提供，不需要未来行情行", "最后已知交易日与其他日一样执行前收盘请求并按真实收盘记账",
                    "最后收盘仍产生下一计划执行意向", "未自然结束的持仓标记观察未完成，不假平仓或计算完成收益"],
        "unchanged": "进入退出条件、模型控制器、费用、登记股息、涨跌停、整手、等待及实际参考份额规则均保留",
        "original_sources_modified": False, "future_price_rows_created": 0,
    })
    print("三种参考程序的完整收盘观察版已生成，旧程序保持原样。", flush=True)


if __name__ == "__main__":
    main()
