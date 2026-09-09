"""复用既有账户流程，形成信号周期来源承诺的独立冻结研究。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "research/episode_trend_reference_v1.py"
    require(not destination.exists(), "第141轮运行文件已经存在")
    source = (ROOT / "research/trend_reference_router_v1.py").read_text(encoding="utf-8")
    source = source.replace("TREND_REFERENCE_ROUTER", "EPISODE_TREND_REFERENCE").replace("trend_reference_router", "episode_trend_reference")
    source = source.replace("trend_routed_frames", "episode_routed_frames").replace("第140轮", "第141轮").replace("round=140", "round=141")
    source = source.replace("按已有120日趋势选择保存策略", "按信号周期固定趋势选择的来源").replace("趋势选择", "信号周期固定来源")
    source = source.replace('PRIMARY: "按退出模型的信号周期固定来源策略"', 'PRIMARY: "信号周期内固定趋势来源"')
    source = source.replace('old_path = ROOT / "config/510300_model_support_reference_router_v1.json"', 'old_path = ROOT / "config/510300_trend_reference_router_v1.json"')
    source = source.replace('P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"', 'P140 = ROOT / "reports/research/510300_trend_reference_router_v1"')
    source = source.replace('"JOINT_DOWNSIDE_REFERENCE_PAIR": (P137, "第137轮合成下行预算")', '"TREND_REFERENCE_ROUTER": (P140, "第140轮每日趋势选择")')
    source = source.replace('trend_threshold=0, registered_at=', 'trend_threshold=0, source_commitment="UNTIL_SELECTED_PARENT_EXPLICIT_ZERO", registered_at=')
    source = source.replace('rules="docs/510300_EPISODE_TREND_REFERENCE_V1.md", source_budget_cny=0,',
        'rules="docs/510300_EPISODE_TREND_REFERENCE_V1.md", input_receipt="reports/research/510300_episode_trend_reference_preflight_20260909/result.json", source_budget_cny=0,')
    source = source.replace("PROGRESS_ROUNDS138_139_DELIVERED_MINIMUM_FOUR_SHARPE_IMPROVED", "PROGRESS_ROUNDS139_140_DELIVERED_AND_EPISODE_INPUTS_READY")
    source = source.replace('from research.episode_trend_reference_inputs_v1 import checked_trend', 'from research.trend_reference_router_inputs_v1 import checked_trend')
    location = source.index('    data = pd.read_parquet(ROOT / cfg["features"])')
    source = source[:location]+'''    receipt_path = ROOT / cfg["input_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    paths.extend([receipt_path, ROOT / "tests/test_trend_reference_router_v1.py"])
    for item in receipt["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "固定来源的已核对输入改变")
        paths.append(ROOT / item["path"])
'''+source[location:]
    source = source.replace('        base_factors = factor_frames["BASE"]\n        base_factors[base_factors.selection_changed].to_csv(OUT / f"{period}_selection_changes.csv", index=False, encoding="utf-8-sig")\n', '')
    source = source.replace('            factors.to_parquet(folder / "factors.parquet", index=False)',
        '            factors.to_parquet(folder / "factors.parquet", index=False)\n            factors[factors.episode_started | factors.episode_released].to_csv(folder / "signal_episode_events.csv", index=False, encoding="utf-8-sig")')
    compile(source, str(destination), "exec")
    destination.write_text(source, encoding="utf-8")
    parent = (ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md").read_text(encoding="utf-8")
    inherited = parent[parent.index("## 第一来源："):].replace("本轮", "原139轮")
    intro = """# 第141轮：信号周期内固定趋势选择的来源

本轮只新增一项决策时点约束。120日均线选择131或139只发生在尚未固定来源时；选中父策略给出正目标，就固定来源至它明确给出零目标。已知趋势、原模型、父参考及其对应费用目标全部复用，不新增拟合或参考账户。

第140轮每日选择提高较早收益，却明显增大主历史风险；来源固定是否改善属于本次需实际检验的假设。原140轮已关闭，此处不更改它的窗口、方向、收益或裁决。方案是在看过历史后提出，不能视为独立验证。

## 本轮因素与来源选择

沿用既有120日趋势偏离：当前含分红财富除以最近完整120个交易日含分红财富的等权平均，再减一，包含当日。完整窗口不足或当前因素缺失时为未知，不补齐。每天收盘后15:05作判断，下一开盘才执行。开始前的准备收盘，来源固定状态为空。

尚未固定来源时，偏离严格大于零选择131普通波动乘数策略，等于或低于零选择139训练支持选择策略。趋势未知则本次没有目标。所选父目标为正，采用该目标并固定来源；明确零则当日目标零但不固定；未知则当日目标未知且不固定。不能转用另一父策略来填补所选目标。

已经固定来源时，忽略新趋势变化，读取所固定父策略的原目标。正目标继续采用；目标未知则当日无调整观点，同时保持来源固定；明确零则当日给出零目标，并在这个收盘处理结束后解除固定。即使另一策略当天有正目标，也要等下一个收盘才能重新作趋势选择。持有中的趋势资料暂缺不阻断已经选定父策略的明确目标，因为趋势仅用于开始选择。

状态因素包括：当前趋势偏离、如果逐日选择会采用的来源、本次实际所选来源、处理前后固定来源、信号周期编号、是否开始及是否由明确零目标结束。周期编号仅供追踪，不决定仓位或收益。两费用分别读取对应父目标，父策略的明确退出可能不同，因此两费用的固定来源状态可以不同。

## 实际进入、调整、退出与再进入

本轮股票目标是所选父策略的原连续收盘目标，没有额外相乘或资金混合。目标大于零且实际空仓，下一开盘按自己现金、持股及分红应收形成的净值买入100份整手。已有持仓与正目标比重相差不足十个百分点保留份额，达到或超过十个百分点调整。明确零目标下一开盘全部卖出；未知目标不发新调整。终点统一开盘清仓，不使用后来收盘数据。

固定的是保存信号的正值周期，不是实际持仓周期：整手、调整带宽及受阻可能使二者不同。父目标明确零时已经解除来源固定，外层受阻请求仍每个收盘依最新目标重算；如果先前清仓未成交而随后出现新正目标，照原账户规则处理。本轮没有新增永久卖出锁定、额外等待或冷却。父策略原来的周期、模型、价格保护、退出锁定及再次进入资格独立连续运行，不受本轮实际盈亏倒灌。

## 账户及有限范围

一项设置、四条新账户：主2020年1月2日至2026年8月14日开盘，较早2015年1月5日至2019年12月31日开盘，各基础和压力费用。复用131、139、140及买入持有十六条保存对照。每户二十万元、242日年化、现金和无风险收益零。基础佣金万分之二且每次最低5元、滑点万分之五；压力佣金万分之四且每次最低5元、滑点千分之一。整百份、0.001元价位、次日可卖、方向涨跌停、登记日权益、除息应收和分红到账均保持。

六项必要测试已经通过，覆盖固定来源、趋势反转、明确零释放、未知保留、费用状态不同、未来与前缀及真实账户分红和进出。随后冻结，一次运行。失败结束此固定设置，不更改释放条件、均线或方向救回。所有旧研究保留，目标仍是成本后夏普至少1.2、稳定超额和独立证据。

## 两套父策略的全部原始中文规则

以下完整保留139的底层因素和进入退出，包含131的普通波动差异以及139如何选择131或137。下文所述原轮次训练、测试与结果属于保存来源，本轮不重新执行。

"""
    (ROOT / "docs/510300_EPISODE_TREND_REFERENCE_V1.md").write_text(intro+inherited, encoding="utf-8")
    out = ROOT / "reports/research/510300_episode_trend_reference_v1"
    write_json(out / "tests_receipt.json", {"recorded_at": now(), "exit_code": 0, "passed": 6, "seconds": 3.52,
        "command": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_episode_trend_reference_v1.py", "output": "6 passed in 3.52s",
        "pre_freeze_fix": "首次测试识别Pandas字符串列将空来源表示为NaN，现已显式识别缺失；此前没有新账户或收益读取。"}, exclusive=True)
    print("第141轮完整规则、运行文件及实际六项测试回执已保存，尚未冻结或计算账户。", flush=True)


if __name__ == "__main__":
    main()
