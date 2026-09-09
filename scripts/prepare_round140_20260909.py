"""复用已核对账户程序，准备完整规则并实际执行必要测试。"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/trend_reference_router_v1.py"
    require(not destination.exists(), "第140轮运行文件已经存在")
    source = (ROOT / "research/model_support_reference_router_v1.py").read_text(encoding="utf-8")
    source = source.replace("model_support_reference_router", "trend_reference_router").replace("MODEL_SUPPORT_REFERENCE_ROUTER", "TREND_REFERENCE_ROUTER")
    source = source.replace("support_routed_frames", "trend_routed_frames").replace("第139轮", "第140轮")
    source = source.replace("按已形成的模型训练支持选择已有策略", "按已有120日趋势选择保存策略").replace("模型训练支持选择", "已有趋势选择").replace("训练支持选择", "趋势选择")
    source = source.replace("按退出模型的训练支持选择策略", "按120日趋势选择已有策略")
    begin, end = source.index("P131 = ROOT"), source.index("\n\ndef freeze():")
    source = source[:begin]+'''P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P139 = ROOT / "reports/research/510300_model_support_reference_router_v1"
P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数策略"), MODELS[1]: (P139, "第139轮训练支持条件选择"),
    "JOINT_DOWNSIDE_REFERENCE_PAIR": (P137, "第137轮合成下行预算"), "BUY_HOLD": (P32, "买入持有")}
'''+source[end:]
    begin, end = source.index("def freeze():"), source.index("\n\ndef run():")
    source = source[:begin]+'''def freeze():
    require(not CONFIG.exists(), "趋势选择已经冻结")
    old_path = ROOT / "config/510300_model_support_reference_router_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {k: old[k] for k in keys}
    old_hashes = {str(Path(v["path"])): v["sha256"] for v in old["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == old_hashes[str(Path(cfg[key]))], "趋势选择行情或分红不再等于既有来源")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "趋势选择六项必要测试未通过")
    cfg.update(study_id="510300_TREND_REFERENCE_ROUTER_V1", round=140, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", trend_window=120, trend_threshold=0, registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_TREND_REFERENCE_ROUTER_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS138_139_DELIVERED_MINIMUM_FOUR_SHARPE_IMPROVED")
    paths = [Path(__file__), old_path, ROOT / "config/510300_vintage_reference_risk_v1.json",
        ROOT / "research/trend_reference_router_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / "config/510300_research_authority_v6.json", ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / cfg["rules"], ROOT / "docs/510300_TREND_REFERENCE_ROUTER_NEXT_20260909.md",
        P131 / "saved_verification_receipt.json", P139 / "saved_verification_receipt.json", P139 / "saved_diagnostic_receipt.json"]
    data = pd.read_parquet(ROOT / cfg["features"])
    from research.trend_reference_router_inputs_v1 import checked_trend
    checked_trend(data)
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            for model, folder in [(MODELS[0], P131), (MODELS[1], P139)]:
                path = folder / period / cost / f"{model}_decisions.parquet"
                parent = pd.read_parquet(path)
                require(np.array_equal(parent.origin_index, indices), "趋势父目标索引不同")
                require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                    pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "趋势父目标时钟不同")
                checks.append({"period": period, "cost": cost, "model": model, "origins": len(parent), "unknown_targets": int(parent.reference_weight.isna().sum())})
                paths.append(path)
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = checks
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第140轮趋势选择已冻结，尚未生成新目标或账户。", flush=True)
'''+source[end:]
    begin = source.index('        source_folders = [(MODELS[0], P131), (MODELS[1], P137)]')
    end = source.index('        parents_by_cost =', begin)
    source = source[:begin]+'''        source_folders = [(MODELS[0], P131), (MODELS[1], P139)]
'''+source[end:]
    source = source.replace('trend_routed_frames(frame, records, parents_by_cost, cfg, start)', 'trend_routed_frames(frame, parents_by_cost, cfg, start)')
    source = source.replace('"训练支持选择": target_summaries', '"趋势选择": target_summaries')
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    parent_text = (ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md").read_text(encoding="utf-8")
    inherited = parent_text[parent_text.index("## 第一来源："):].replace("本轮", "原139轮")
    intro = """# 第140轮：已有趋势因素选择两套保存策略

每日15:05，含分红财富严格高于最近完整120个交易日均值，采用131普通波动乘数策略；等于或低于均值，采用139训练支持选择策略。该规则属于看过历史后提出的新固定组合，没有独立验证。139的训练门槛和原结果不改变。

## 本轮新增选择因素、进入与退出

完整使用原行情中已有的一百二十日均线偏离：当日含分红财富除以最近一百二十个交易日含分红财富的等权平均，再减一，窗口含当日。核对保存因素与此定义一致。因素严格大于零选131；等于或小于零选139。趋势因素缺失、完整窗口不足或所选目标未知，保持无观点，不按另一父策略补齐。没有搜索其他均线、方向、阈值或等待天数。均线可用于描述趋势的概念参见[CME均线说明](https://www.cmegroup.com/education/courses/technical-analysis/understanding-moving-averages)；该资料不证明本组合能盈利或达到目标。

两费用共用趋势状态，目标必须读取对应费用父策略已经保存的收盘判断。131与139的内部账户、训练、等待新机会和退出锁定继续按原路径；本轮不重新训练，不重跑参考，不将自己的盈亏或持股倒灌到父策略。139内部再次选择131或137是原规则，不是本轮额外调参。

所选目标大于零、外层空仓时，下一开盘按自身净值买入整百份。已有持股时，正目标与实际比重相差不足十个百分点保留份额，达到十个百分点增减仓；明确零目标在下一开盘全部退出，未知目标不发新调整。受阻请求每个收盘重新按最新目标计算。全部卖完后，新正目标可再次进入，无额外冷却或永久退出锁定。终点统一开盘清仓，不使用终点后来才形成的收盘因素。上述为实际外层进出规则，下文完整列出父策略因素和进出规则。

一项设置、四条新账户；复用131、139、137和买入持有十六条对照。主2020年1月2日至2026年8月14日开盘，较早2015年1月5日至2019年12月31日开盘。两段均两档费用、二十万元、242日年化及零现金收益，保留分红、整手、涨跌停和次日可卖。必要测试后冻结，一次运行；失败结束此固定组合，不更改窗口或方向救回。

## 两套父策略的全部原始中文规则

以下是已保存第139轮的完整底层规则，涉及原轮次的测试、训练和结果均是历史记录。本轮读取131与139的最终收盘目标；其中131的差异及139的训练支持选择均在下文具体说明。

"""
    (ROOT / "docs/510300_TREND_REFERENCE_ROUTER_V1.md").write_text(intro+inherited, encoding="utf-8")
    out = ROOT / "reports/research/510300_trend_reference_router_v1"
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_trend_reference_router_v1.py"]
    began = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = process.stdout+process.stderr
    (out / "tests_output.txt").write_text(output, encoding="utf-8")
    require(process.returncode == 0 and re.search(r"\b6 passed\b", output) is not None, "趋势选择必要测试失败，详见tests_output.txt")
    write_json(out / "tests_receipt.json", {"tested_at": now(), "exit_code": process.returncode, "passed": 6,
        "wall_seconds": time.perf_counter()-began, "command": command, "output": output}, exclusive=True)
    print(json.dumps({"测试": output.strip(), "规则": "docs/510300_TREND_REFERENCE_ROUTER_V1.md", "账户尚未运行": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
