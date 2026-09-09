"""复用已验证的单设置完整账户运行框架，生成第119轮独立完整入口。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "research/market_path_exit_v1.py").read_text(encoding="utf-8")
replacements = {
    '单一条件不含已付买入费用的八项周期内退出模型的完整历史账户检验。': '单一联合周期截距胡伯目标的完整历史账户检验。',
    'from research.market_path_exit_inputs_v1 import FEATURES, CN, fit_market_path_exit, MarketPathExitController': 'from research.robust_cycle_exit_inputs_v1 import FEATURES, CN, fit_robust_cycle, RobustCycleExitController',
    '"reports/research/510300_market_path_exit_v1"': '"reports/research/510300_robust_cycle_exit_v1"',
    '"config/510300_market_path_exit_v1.json"': '"config/510300_robust_cycle_exit_v1.json"',
    '"MARKET_PATH_EXIT"': '"ROBUST_CYCLE_EXIT"',
    '"510300_MARKET_PATH_EXIT_V1"': '"510300_ROBUST_CYCLE_EXIT_V1"',
    '"docs/510300_MARKET_PATH_EXIT_V1.md"': '"docs/510300_ROBUST_CYCLE_EXIT_V1.md"',
    '"tests/test_market_path_exit_v1.py"': '"tests/test_robust_cycle_exit_v1.py"',
    '"research/market_path_exit_inputs_v1.py"': '"research/robust_cycle_exit_inputs_v1.py"',
    'round=117': 'round=119',
    '第117轮': '第119轮',
    '不含已付买入费用的八项周期内': '联合周期截距胡伯',
    'solver="svd"': 'solver="L-BFGS-B"',
    '"WITHIN_CYCLE_CENTERED_SQUARE_ERROR_WITH_BETA_RIDGE_AND_UNPENALIZED_CYCLE_INTERCEPTS"': '"JOINT_CYCLE_HUBER_WITH_BETA_RIDGE_AND_UNPENALIZED_CYCLE_INTERCEPTS"',
    'added_feature_columns=FEATURES[1:3], dropped_feature_columns=["cycle_return", "cycle_drawdown"]': 'added_feature_columns=[], dropped_feature_columns=[]',
    '"PROGRESS_ROUND116_COMPLETED_AND_SUNK_COST_STATE_IDENTIFIED"': '"PROGRESS_ROUND118_COMPLETED_AND_ROBUST_SCALE_CHECKED"',
    'MarketPathExitController(frame, models, cfg["confirmation_days"])': 'RobustCycleExitController(frame, models, cfg["confirmation_days"])',
    'stored = fit_market_path_exit(rows, cfg)': 'require(initials[t]["status"] == "FIT_COMPLETE" and initials[t]["training_cycles"] == ids and initials[t]["training_rows"] == len(rows), "联合胡伯初始化不是同月同一成熟周期")\n                stored = fit_robust_cycle(rows, cfg, initials[t]["model"])',
    '"MARKET_PATH_EXIT_ACCOUNTS_COMPLETE"': '"ROBUST_CYCLE_EXIT_ACCOUNTS_COMPLETE"',
}
for old, new in replacements.items():
    if old not in source:
        raise ValueError("复用框架的预期位置不存在："+old)
    source = source.replace(old, new)
anchor = '    paths = [Path(__file__)'
if source.count(anchor) != 1:
    raise ValueError("框架冻结路径位置不唯一")
source = source.replace(anchor, '''    cfg.update(initial_models="reports/research/510300_market_path_exit_v1/saved_models.json",
        residual_preflight_receipt="reports/research/510300_robust_cycle_scale_preflight_20260908/result.json",
        huber_constant=1.345, maximum_iterations=1000, objective_tolerance=1e-15, gradient_tolerance=1e-9,
        maximum_line_search_steps=40, accepted_gradient_bound=1e-7, residual_mad_normalization=0.6744897501960817,
        residual_scale="FIXED_WEIGHTED_MAD_OF_SAME_MONTH_ORIGINAL_CYCLE_SPECIFIC_RESIDUALS")
    paths = [Path(__file__)''')
anchor = '    for period in ["evaluation", "earlier_diagnostic"]:'
if source.count(anchor) != 1:
    raise ValueError("保存对照路径位置不唯一")
source = source.replace(anchor, '''    paths.extend([ROOT / cfg["initial_models"], ROOT / cfg["residual_preflight_receipt"], ROOT / "research/market_path_exit_inputs_v1.py",
        ROOT / "reports/research/510300_finite_horizon_exit_v1/saved_verification_receipt.json"])
    for period in ["evaluation", "earlier_diagnostic"]:''')
anchor = '    coefficients, cycle_effects = [], []'
source = source.replace(anchor, '''    coefficients, cycle_effects = [], []
    initial_records = json.loads((ROOT / cfg["initial_models"]).read_text(encoding="utf-8"))["models"]
    require([r["fit_index"] for r in initial_records] == [r["fit_index"] for r in originals], "联合胡伯初始模型与原月度时钟不同")
    initials = {r["fit_index"]: r for r in initial_records}''')
target = ROOT / "research/robust_cycle_exit_v1.py"
with target.open("x", encoding="utf-8", newline="\n") as output:
    output.write(source)
compile(source, str(target), "exec")
print("第119轮独立完整账户入口已生成并通过语法编译；尚未冻结、拟合或回测。")
