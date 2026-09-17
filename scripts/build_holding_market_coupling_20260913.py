"""复用已固定的时钟和模拟账户结构，生成本轮独立的联合作用模型实现。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(name, text):
    with (ROOT / name).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    original = (ROOT / 'research/single_component_exit_inputs_v1.py').read_text(encoding='utf-8')
    prefix = '''"""以市场状态调节持仓状态的边际退出作用，按成熟输入向前缓存。"""
import copy
import hashlib
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "WITHIN_CYCLE_HOLDING_MARKET_COUPLING_RIDGE"
PAIRS = [(i, j) for i in range(3) for j in range(4, 8)]
NAMES = FEATURES + [FEATURES[i]+"_times_"+FEATURES[j] for i,j in PAIRS]
CHINESE = CN + [CN[i]+"与"+CN[j]+"的联合作用" for i,j in PAIRS]
IDENTITY_KEYS = ["kind", "features", "mean", "scale", "coefficients", "intercept", "feature_clip",
                 "interaction_pairs", "product_mean", "product_scale"]


def standardize(x, w):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    return np.clip((x-mean)/scale, -5., 5.), mean, scale


def transformed(model, values):
    values = np.asarray(values, float)
    require(values.shape[-1] == 8 and np.isfinite(values).all(), "预测需要完整八项因素")
    z = np.clip((values-model["mean"])/model["scale"], -5., 5.)
    products = np.stack([z[...,i]*z[...,j] for i,j in PAIRS], axis=-1)
    joined = np.clip((products-model["product_mean"])/model["product_scale"], -5., 5.)
    return np.concatenate([z, joined], axis=-1)


def fit_coupling_cycle(rows, cfg):
    require(cfg["feature_clip"] == 5. and cfg["ridge_alpha"] == 1., "固定尺度或惩罚改变")
    require(cfg["interaction_pairs"] == [list(p) for p in PAIRS], "十二项联合作用身份不同")
    x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows)>0 and np.isfinite(x).all() and np.isfinite(y).all(), "训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w>0).all(), "训练权重无效")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "训练周期原点重复")
    require((rows.origin_index<rows.exit_index).all(), "训练状态必须早于自然退出")
    z, mean, scale = standardize(x, w)
    products = np.stack([z[:,i]*z[:,j] for i,j in PAIRS], axis=1)
    joint, product_mean, product_scale = standardize(products, w)
    design = np.column_stack([z, joint])
    dx, dy, groups = np.empty_like(design), np.empty_like(y), []
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        require(np.allclose(w[mask], 1./mask.sum(), atol=1e-14, rtol=0), "每周期总权重必须为一且状态等权")
        mz, my = np.average(design[mask], axis=0, weights=w[mask]), float(np.average(y[mask], weights=w[mask]))
        dx[mask], dy[mask] = design[mask]-mz, y[mask]-my
        groups.append({"cycle_id": int(cycle), "rows": int(mask.sum()), "standardized_feature_mean": mz.tolist(), "target_mean": my})
    fit = Ridge(alpha=1., solver="svd", fit_intercept=False).fit(dx, dy, sample_weight=w)
    beta = fit.coef_
    require(np.isfinite(beta).all(), "联合作用系数非有限")
    residual = dx@beta-dy
    normal_error = float(np.max(abs(dx.T@(w*residual)+beta)))
    require(normal_error<1e-8, "岭回归正规方程未满足")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"]-np.array(group["standardized_feature_mean"])@beta)
    return {"kind": KIND, "features": FEATURES.copy(), "expanded_features": NAMES.copy(), "mean": mean.tolist(),
        "scale": scale.tolist(), "product_mean": product_mean.tolist(), "product_scale": product_scale.tolist(),
        "interaction_pairs": [list(p) for p in PAIRS], "coefficients": beta.tolist(),
        "intercept": float(np.mean([g["cycle_intercept"] for g in groups])), "feature_clip": 5.,
        "cycle_intercepts": groups, "max_normal_error": normal_error,
        "nonzero_factors": [name for name,b in zip(NAMES,beta) if b!=0.], "nonzero_factor_count": int((beta!=0.).sum()),
        "training_weighted_mse": float(np.average(residual**2, weights=w)),
        "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def input_identity(rows, cfg):
    settings = {"features": FEATURES, **{key: cfg[key] for key in ["feature_clip", "ridge_alpha", "interaction_pairs"]}}
    h = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    h.update(rows[["cycle_id", "origin_index", "exit_index"]].to_numpy(dtype="<i8").tobytes())
    h.update(rows[FEATURES+["target", "sample_weight"]].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    content = {key:record["model"][key] for key in IDENTITY_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


'''
    cache = original[original.index('def build_monthly_models('):original.index('def single_component_prediction(')]
    cache = cache.replace('fit_single_component_cycle', 'fit_coupling_cycle')
    cache = cache.replace('"score_residual_covariance": model["score_residual_covariance"] if model else None,',
                          '"max_normal_error": model["max_normal_error"] if model else None,')
    cache = cache.replace('"zero_covariance_model": model["zero_covariance_model"] if model else None',
                          '"zero_coefficients": model["nonzero_factor_count"] == 0 if model else None')
    prediction = '''def coupling_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "联合作用退出模型身份不同")
    require(model["interaction_pairs"] == [list(p) for p in PAIRS], "联合作用预测次序不同")
    return float(model["intercept"]+transformed(model, values)@np.array(model["coefficients"]))


'''
    controller = original[original.index('class SingleComponentExitController:'):]
    controller = controller.replace('SingleComponentExitController', 'CouplingExitController').replace('single_component_prediction', 'coupling_prediction').replace('单成分', '联合作用')
    save('research/holding_market_coupling_exit_inputs_v1.py', prefix+cache+prediction+controller)

    old_runner = (ROOT/'research/single_component_exit_v1.py').read_text(encoding='utf-8')
    run = old_runner[old_runner.index('def run():'):old_runner.index('if __name__ == "__main__":')]
    run = run.replace('SingleComponentExitController', 'CouplingExitController').replace('单成分', '联合作用')
    run = run.replace('原八因素合成单一分数后固定版本退出', '市场状态调节持仓退出')
    run = run.replace('SINGLE_COMPONENT_EXIT_ACCOUNTS_COMPLETE', 'HOLDING_MARKET_COUPLING_EXIT_ACCOUNTS_COMPLETE')
    run = run.replace('"reused_control_accounts": 8', '"reused_control_accounts": 6').replace('"reused_earlier_accounts": 8', '"reused_earlier_accounts": 6')
    run = run.replace('和四个保存对照', '和三个保存对照')
    run = run.replace('    write_json(OUT / "result.json", result, exclusive=True)',
                      '    write_json(OUT / "result.json", result, exclusive=True)\n    save_joint_assessment(OUT, cfg, result)')
    header = '''"""第192轮固定二十项周期内模型和四条完整模拟账户。"""
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.holding_market_coupling_exit_inputs_v1 import build_monthly_models, CouplingExitController, PAIRS
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules
from research.joint_account_acceptance_v1 import save_joint_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_holding_market_coupling_exit_v1'
CONFIG = ROOT/'config/510300_holding_market_coupling_exit_v1.json'
PRIMARY = 'HOLDING_MARKET_COUPLING_EXIT'
CONTROLS = {
    'ENTRY_VINTAGE_EXIT': (ROOT/'reports/research/510300_entry_vintage_exit_v1', '第128轮原固定版本退出'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '第181轮原账户风险预算'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本轮已冻结或计算')
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT/'tests_receipt.json').exists(), '本轮已有必要测试回执')
    started = time.perf_counter()
    tested = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_holding_market_coupling_exit_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr, encoding='utf-8')
    print(tested.stdout, flush=True)
    require(tested.returncode==0 and '6 passed' in tested.stdout, '本轮六项必要测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': tested.returncode, 'passed': 6,
        'seconds': time.perf_counter()-started, 'timing_scope': '完整测试进程墙钟'}, exclusive=True)
    parent_path = ROOT/'config/510300_within_cycle_exit_v1.json'
    old = json.loads(parent_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal',
        'confirmation_days', 'specification', 'feature_columns', 'feature_names', 'feature_clip', 'ridge_alpha',
        'recent_cycles', 'minimum_cycles', 'minimum_rows']
    cfg = {k:old[k] for k in keys}
    cfg.update(study_id='510300_HOLDING_MARKET_COUPLING_EXIT_V1', round=192, registered_at=now(), primary=PRIMARY,
        candidate_models=[PRIMARY], candidate_configurations=1, annual_return_target=.10,
        source_models='reports/research/510300_within_cycle_exit_v1/saved_models.json',
        samples='reports/research/510300_within_cycle_exit_v1/extended_reference_samples.parquet',
        interaction_pairs=[list(p) for p in PAIRS], total_regressors=20, planned_distinct_fits=25,
        planned_monthly_records=141, planned_eligible_months=114, planned_reused_monthly_fits=89,
        model_selection_clock='FIRST_CLOSE_OF_ACTUAL_FILLED_ENTRY', model_reselection='ONLY_ON_NEW_ACTUAL_CYCLE',
        rules='docs/510300_HOLDING_MARKET_COUPLING_EXIT_V1.md', new_reference_accounts=0, source_budget_cny=0,
        goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        previous_goal_turn_classification='PROGRESS_ROUND191_VERIFIED_DELIVERED_AND_INDEX_UPDATED')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        rules=(ROOT/'docs/510300_HOLDING_MARKET_COUPLING_EXIT_NEXT_20260913.md').read_text(encoding='utf-8')
        stream.write(rules.replace('本方案尚未冻结、拟合或计算账户。', '本方案在必要测试通过后、首次新模型拟合和账户计算前冻结。'))
    paths=[Path(__file__), parent_path, ROOT/'research/holding_market_coupling_exit_inputs_v1.py',
        ROOT/'research/learned_cycle_exit_v1.py', ROOT/'research/rearmed_cycle_exit_account_v1.py',
        ROOT/'research/simple_intraday_protection_v1.py', ROOT/'research/simple_session_divergence_v1.py',
        ROOT/'research/simple_price_entry_exit_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'tests/test_holding_market_coupling_exit_v1.py', ROOT/'tests/test_median_continuation_v1.py',
        ROOT/'tests/test_single_component_exit_v1.py', OUT/'tests_receipt.json', OUT/'tests_output.txt',
        ROOT/'config/510300_research_authority_v6.json']
    paths += [ROOT/cfg[k] for k in ['rules', 'features', 'dividends', 'samples', 'source_models']]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)), 'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==191, '本轮前序完成状态不同')
    index['running_studies']=[{'round':192,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='HOLDING_MARKET_COUPLING_EXIT_FROZEN',source=cfg['rules'])
    write_json(path,index)
    print('第192轮固定方案已冻结，尚未计算新历史模型或账户。',flush=True)


'''
    save('research/holding_market_coupling_exit_v1.py', header+run+'''if __name__ == '__main__':
    {'prepare':prepare, 'run':run}[sys.argv[1]]()
''')
    print('本轮独立模型和账户入口已生成；尚未训练或回测。')


if __name__ == '__main__':
    main()
