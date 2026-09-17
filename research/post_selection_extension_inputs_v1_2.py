"""接纳已保存行情，核对因子前缀，并裁剪固定两来源方案的续算依赖。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import factors
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_post_selection_extension_inputs_v1'
CONFIG = ROOT / 'config/510300_post_selection_extension_inputs_v1_2.json'
PREVIOUS = ROOT / 'reports/research/510300_post_selection_data_feasibility_v1'
PRIMARY = 'SELECTED_MIX_BAND10_SIMPLE2'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def node(key, study, parents, entry, kind, state, rule, folder=None, start='2020-01-02'):
    """区分来源目标、实际账户和模型资料，避免把旧对照误列为依赖。"""
    cfg = 'config/510300_' + study + '_v1.json'
    program, function = entry.split(':')
    return dict(node=key, configuration=cfg, direct_parents=parents,
        program='research/' + program + '.py', entry_function=function,
        output_kind=kind, required_state_cn=state, rule_cn=rule,
        saved_folder=folder or 'reports/research/510300_' + study + '_v1',
        prices='candidate_prices.parquet', factors='candidate_features.parquet',
        dividends='data/reference/510300_dividends.csv', replay_start=start,
        complete_checkpoint_verified=False, restoration='原纯函数重算因素；仅必要账户重放并增加可恢复状态')


def dependencies():
    nodes = [
        node(PRIMARY, 'incremental_selected_intent_mix', ['ADD_GATE_EXPOSURE_115', 'RUNS_OPPORTUNITY_CAPPED_SUM'],
             'incremental_selected_intent_mix_inputs_v1:simulate_gate_account', 'ACCOUNT_BOTH_COSTS', '现金、份额、分红权利、待执行申请',
             '按两个来源计划持有比例的85%和15%合成目标；十个百分点调仓门槛；明确零全部退出；不重复加仓过滤。'),
        node('ADD_GATE_EXPOSURE_115', 'addition_gate_exposure_batch', ['INTENT_MIX_BAND_20'],
             'addition_gate_exposure_batch_inputs_v1:simulate_gate_account', 'ACCOUNT_BOTH_COSTS', '普通账户状态；已有持仓加仓许可',
             '目标乘1.15后封顶100%；二十个百分点调仓门槛；已有持仓仅当当日含分红收益达到1%才加仓；初始买入和卖出保持原规则。'),
        node('INTENT_MIX_BAND_20', 'finite_rebalance_band_batch', ['INTENT_MIX_80_15_05'],
             'finite_rebalance_band_batch_inputs_v1:band_frames', 'TARGET_ONLY', '无需该层实际账户',
             '供206读取的原始目标等于200目标；该层实际成交不参与206计算，因此不重放201账户。'),
        node('INTENT_MIX_80_15_05', 'three_source_order_intent_mix', ['TWO_CLOSE_ZERO_EXIT', 'CONFIRMED_RISK_EPISODE_30_10', 'MEAN_REBOUND_AUX_50'],
             'three_source_order_intent_mix_inputs_v1:source_plan', 'TARGET_ONLY', '三个父账户的份额、净值和当日申请',
             '以三个父来源计划持有比例按80%、15%、5%合并；未知传播；不能用原始目标代替计划份额。'),
        node('TWO_CLOSE_ZERO_EXIT', 'two_close_zero_exit', ['ACCOUNT_VOLATILITY_EXPOSURE'],
             'two_close_zero_exit_inputs_v1:confirmation_request', 'ACCOUNT_BOTH_COSTS', '账户状态、连续明确零计数',
             '首次明确零保持份额；连续第二次及之后零全部退出；正目标或未知重置计数；正目标按十个百分点调仓。'),
        node('CONFIRMED_RISK_EPISODE_30_10', 'confirmed_exit_existing_budget_batch', ['RISK_EPISODE_30_10'],
             'confirmed_exit_existing_budget_batch_inputs_v1:simulate_mapped_confirmation', 'ACCOUNT_BOTH_COSTS', '账户状态、连续明确零计数',
             '只用三十日区间固定10%风险预算来源，并在该来源外执行连续两次明确零退出。'),
        node('MEAN_REBOUND_AUX_50', 'mean_rebound_auxiliary', ['ACCOUNT_VOLATILITY_EXPOSURE', 'R2_Z_CONFIRM'],
             'mean_rebound_auxiliary_inputs_v1:rebound_frames', 'ACCOUNT_BOTH_COSTS', '账户状态；两个父目标',
             '181原目标加上一半均值反弹目标，封顶100%；任一来源未知则未知；十个百分点调仓、明确零退出。'),
        node('ACCOUNT_VOLATILITY_EXPOSURE', 'account_volatility_exposure', ['EITHER_CONFIRMED_RUNS_AUXILIARY'],
             'account_volatility_exposure_inputs_v1:risk_budget', 'TARGET_ONLY', '174的最近六十日实际净收益窗口',
             '按174实际账户六十日波动把目标缩放到10%年化风险，封顶100%；准备期和零波动按原规则；该层无需重放账户。'),
        node('RISK_EPISODE_30_10', 'risk_window_clock_batch', ['EITHER_CONFIRMED_RUNS_AUXILIARY'],
             'episode_account_risk_budget_inputs_v1:fixed_episode_budget', 'TARGET_ONLY', '174的三十日实际收益窗口、正目标区间起点和固定乘数',
             '正目标区间开始时，以三十日实际账户波动确定10%风险预算并固定乘数；来源目标每日变化仍保留。'),
        node('EITHER_CONFIRMED_RUNS_AUXILIARY', 'return_confirmation_auxiliary_batch', ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'RETURN_LAG_STATE', 'RETURN_SIGN_BALANCE'],
             'return_confirmation_auxiliary_batch_inputs_v1:return_confirmation_auxiliary_batch_frames', 'ACCOUNT_BOTH_COSTS', '账户实际净收益；各父目标与方向',
             '核心目标加连续段辅助；相邻收益相关方向或涨跌日数方向任一允许时才接纳辅助，封顶100%；所需因素未知则未知。'),
        node('RUNS_OPPORTUNITY_CAPPED_SUM', 'runs_opportunity_union', ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE'],
             'runs_opportunity_union_inputs_v1:runs_opportunity_union_frames', 'ACCOUNT_BOTH_COSTS', '账户状态',
             '核心目标与收益连续段目标相加封顶100%；任一未知则未知；十个百分点调仓；明确零退出。'),
        node('RETURN_RUNS_STATE', 'return_runs_state', [], 'return_runs_state_inputs_v1:return_runs_state_factors', 'FACTOR_TARGET_ONLY',
             '六十日完整收益、进入计数、两个退出计数、方向状态',
             '六十日收益相对窗口中位数的连续段分数低于负一且六十日动量为正，连续两次进入；连续两次分数非负或连续两次动量非正退出；仓位受二十日10%波动预算限制。'),
        node('RETURN_LAG_STATE', 'return_lag_state', [], 'return_lag_state_inputs_v1:return_lag_state_factors', 'FACTOR_DIRECTION_ONLY',
             '六十日收益、进入计数、相关与动量退出计数、方向',
             '六十日窗口中59组相邻收益的相关为正且动量为正连续两次允许；相关非正或动量非正分别连续两次退出；缺失或常数相关未定义时重置。'),
        node('RETURN_SIGN_BALANCE', 'return_sign_balance', [], 'return_sign_balance_inputs_v1:return_sign_balance_factors', 'FACTOR_DIRECTION_ONLY',
             '六十日收益、进入计数、日数与动量退出计数、方向',
             '六十日上涨日数减下跌日数为正且动量为正连续两次允许；日数差非正或动量非正分别连续两次退出；平盘保留在分母。'),
        node('TREND_NOISE_REFERENCE_BLEND', 'trend_noise_reference_blend', ['VINTAGE_REFERENCE_RISK', 'MODEL_SUPPORT_REFERENCE_ROUTER'],
             'trend_noise_reference_blend_inputs_v1:trend_noise_frames', 'TARGET_ONLY', '两个父目标、原趋势与波动因素',
             '一百二十日正趋势幅度除以正趋势与同尺度二十日波动幅度之和，分配131预算，余下分配139；零正趋势时131预算为零。'),
        node('VINTAGE_REFERENCE_RISK', 'vintage_reference_risk', ['ENTRY_VINTAGE_EXIT', 'REALIZED_VOLATILITY20_MULTIPLIER'],
             'vintage_reference_risk_inputs_v1:vintage_risk_targets', 'TARGET_ONLY', '128的费用匹配零一意向、普通波动乘数',
             '固定入场模型参考意向乘以普通二十日波动风险乘数；不依赖条件方差拟合。'),
        node('REALIZED_VOLATILITY20_MULTIPLIER', 'conditional_variance_budget', [],
             'conditional_variance_budget_inputs_v1:budget_frame', 'FACTOR_ONLY', '最近一次有效普通波动乘数，初始为一',
             '仅提取普通波动分支：二十日波动有效且为正时取一与10%除以波动的较小值；否则沿用最近乘数；不需要GARCH、109账户或其父账户。'),
        node('MODEL_SUPPORT_REFERENCE_ROUTER', 'model_support_reference_router', ['VINTAGE_REFERENCE_RISK', 'JOINT_DOWNSIDE_REFERENCE_PAIR', 'WITHIN_CYCLE_SAVED_MODELS'],
             'model_support_reference_router_inputs_v1:support_routed_frames', 'TARGET_ONLY', '15:05之前最近可用模型支持记录',
             '最近已形成模型记录满足原十个成熟周期及一百行要求时取137，否则取131；不按后续表现选择。'),
        node('JOINT_DOWNSIDE_REFERENCE_PAIR', 'joint_downside_reference_pair', ['DOWNSIDE_REFERENCE_RISK', 'CONTINUOUS_REFERENCE_MIN_VARIANCE'],
             'joint_downside_reference_pair_inputs_v1:joint_downside_reference_frames', 'TARGET_ONLY', '两父基础账户242日净收益、上次预算及月首更新时点',
             '月首完整收盘只用过去242日两基础账户收益，最小化组合负收益平方；再合成对应费用的目标；不足窗口保持旧预算。'),
        node('DOWNSIDE_REFERENCE_RISK', 'downside_reference_risk', ['ENTRY_VINTAGE_EXIT'],
             'downside_reference_risk_inputs_v1:moment_reference_targets', 'ACCOUNT_BASE_TARGET_STRESS', '基础账户净收益、费用匹配128意向及下行风险乘数',
             '二十日负收益平方均值乘二再年化求风险，意向乘以不超过一的10%风险预算；压力仅需目标，基础账户供137风险使用。'),
        node('ENTRY_VINTAGE_EXIT', 'entry_vintage_exit', ['WITHIN_CYCLE_SAVED_MODELS'],
             'rearmed_cycle_exit_account_v1:simulate_rearmed_exit', 'ACCOUNT_BOTH_COSTS',
             '实际持仓周期、买入成本、周期高点和分红、退出等待、再入场许可、冷却日、首次买入模型身份、负预测计数',
             '六十日日内强于隔夜信号入场；原价格退出、6%止损、8%追踪、最长六十日及连续两次负继续收益退出；每次实际买入首次收盘固定模型。'),
        node('CONTINUOUS_REFERENCE_MIN_VARIANCE', 'continuous_reference_min_variance', ['PANIC_ONLY', 'REARM_RIDGE_CONTINUOUS'],
             'continuous_reference_min_variance_v1:reference_factors', 'ACCOUNT_BASE_TARGET_STRESS', '连续参考242日收益、方向和月首预算、基础实际账户',
             '两个2013年起连续基础参考按月首已知242日历史更新最小方差预算；合成目标两费用相同，但137需要该策略基础账户的实际收益。'),
        node('PANIC_ONLY', 'continuous_reference_min_variance', [], 'simple_price_entry_exit_v1:simulate_policy', 'ACCOUNT_BASE_ONLY',
             '持仓周期、退出等待、冷却日、成本及周期分红',
             '五日跌幅超过5%后首日上涨、收盘位置至少60%、五日波动大于六十日1.5倍时进入；均值回归、4%止损、6%止盈或十日到期退出。',
             'reports/research/510300_continuous_reference_min_variance_v1/continuous_references', '2013-06-03'),
        node('REARM_RIDGE_CONTINUOUS', 'continuous_reference_min_variance', ['RIDGE_SAVED_MODELS'],
             'rearmed_cycle_exit_account_v1:simulate_rearmed_exit', 'ACCOUNT_BASE_ONLY',
             '持仓周期与退出等待、再入场许可、冷却日、周期分红与高点、最近模型及负预测计数',
             '六十日日内强于隔夜入场和原价格风险退出，连续两次模型预测继续收益为负退出；每天使用当时已可用最近模型。',
             'reports/research/510300_continuous_reference_min_variance_v1/continuous_references', '2013-06-03'),
        node('R2_Z_CONFIRM', 'simple_price_entry_exit', [], 'simple_price_entry_exit_v1:simulate_policy', 'ACCOUNT_BOTH_COSTS',
             '实际持仓周期、成本、分红、退出等待和冷却日',
             '二十日标准化偏离低于负1.5后首日上涨进入；偏离回到零、5%止损或十日到期退出；退出后一日等待。',
             'reports/research/510300_mean_rebound_auxiliary_v1/reference_sources'),
        node('WITHIN_CYCLE_SAVED_MODELS', 'within_cycle_exit', [], 'within_cycle_exit_inputs_v1:within_cycle_prediction', 'SAVED_MODEL_ARTIFACT',
             '141条原模型支持与系数记录、选择时点；最后模型为2026年8月3日',
             '八项原持仓内退出因子；本项只接纳旧模型文件，不拟合；接续到九月时须区分固定旧模型诊断和按原月首规则新增成熟训练记录。'),
        node('RIDGE_SAVED_MODELS', 'learned_cycle_exit', [], 'learned_cycle_exit_v1:predict', 'SAVED_MODEL_ARTIFACT',
             '仅六十日日内来源的141条岭回归记录，最后模型为2026年8月3日',
             '使用原八项持仓状态因子及已保存系数，不扩展到树或其他信号；九月月首模型尚未生成。'),
    ]
    lookup = {n['node']: n for n in nodes}
    for n in nodes:
        require((ROOT/n['configuration']).is_file() and (ROOT/n['program']).is_file(), '依赖配置或程序不存在：'+n['node'])
        source = (ROOT/n['program']).read_text(encoding='utf-8')
        require(('def '+n['entry_function']+'(') in source, '依赖纯入口名称不同：'+n['node'])
        require(all(p in lookup for p in n['direct_parents']), '依赖图存在悬空父节点')
    ordered, active, visited = [], set(), set()
    def visit(key):
        require(key not in active, '依赖图存在循环')
        if key in visited:
            return
        active.add(key)
        for parent in lookup[key]['direct_parents']:
            visit(parent)
        active.remove(key)
        visited.add(key)
        ordered.append(key)
    visit(PRIMARY)
    require(len(visited)==len(nodes), '依赖图包含不被两来源使用的节点')
    return [lookup[key] for key in ordered]


def source_inventory(nodes):
    rows = []
    for n in nodes:
        if not n['output_kind'].startswith('ACCOUNT_'):
            continue
        costs = ['BASE','STRESS'] if n['output_kind']=='ACCOUNT_BOTH_COSTS' else ['BASE']
        for cost in costs:
            model = 'REARM_RIDGE' if n['node']=='REARM_RIDGE_CONTINUOUS' else n['node']
            folder = ROOT/n['saved_folder']
            if n['replay_start']!='2013-06-03':
                folder = folder/'evaluation'
            ledger_path, decisions_path = folder/cost/(model+'_ledger.parquet'), folder/cost/(model+'_decisions.parquet')
            require(ledger_path.is_file() and decisions_path.is_file(), '必要来源账本或决定不存在：'+n['node'])
            ledger, decisions = pd.read_parquet(ledger_path), pd.read_parquet(decisions_path)
            before, last, request = ledger.iloc[-2], ledger.iloc[-1], decisions.iloc[-1]
            require(before.date==pd.Timestamp('2026-08-13') and last.date==pd.Timestamp('2026-08-14'), '来源终点不一致')
            rows.append(dict(node=n['node'], cost=cost, ledger=str(ledger_path.relative_to(ROOT)), decisions=str(decisions_path.relative_to(ROOT)),
                ledger_sha256=digest(ledger_path), decisions_sha256=digest(decisions_path), rows=len(ledger),
                replay_start=str(ledger.date.iloc[0].date()), last_normal_date=str(before.date.date()),
                shares_before_terminal=int(before.shares), cash_before_terminal=float(before.cash),
                receivable_before_terminal=float(before.dividend_receivable), normal_request=int(request.requested_quantity),
                terminal_fill=int(last.filled_quantity), complete_checkpoint_verified=False,
                restoration='先在新目录重放必要来源的原区间，保存原终点强制清仓之前的内部状态，再核对正常区间'))
    return rows


def main():
    began=time.perf_counter()
    require(OUT.is_dir() and {p.name for p in OUT.iterdir()}=={'source_precision_comparison.csv'} and not CONFIG.exists(), '类型修复仅接续精度比较已完成而候选行情未写入的本项')
    index_path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=read(index_path)
    require(index['latest_completed_round']['round']==211 and len(index['running_studies'])==1 and index['running_studies'][0]['round']==212, '接续前序状态不同')
    nodes=dependencies()
    inventory=source_inventory(nodes)
    old_cfg=read('config/510300_incremental_selected_intent_mix_v1.json')
    paths=[ROOT/'data/raw/market/510300_daily_downside_risk_v1.parquet', ROOT/old_cfg['features'], ROOT/old_cfg['dividends'],
        ROOT/'data/reference/510300_dividends_coverage.json',ROOT/'data/reference/sse_trade_calendar_2026.csv', Path(__file__),
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'docs/510300_POST_SELECTION_EXTENSION_INPUTS_NEXT_20260913.md']
    paths.extend(PREVIOUS/name for name in ['sina_daily.parquet','tencent_daily.parquet','sina_receipt.json','tencent_receipt.json',
        'sina.raw','tencent.raw','dividend_candidate_coverage.json','time_boundaries.json','result.json','saved_verification_receipt.json'])
    paths.extend(ROOT/n[k] for n in nodes for k in ['configuration','program'])
    model_paths=['reports/research/510300_within_cycle_exit_v1/saved_models.json','reports/research/510300_learned_cycle_exit_v1/saved_models.json']
    paths.extend(ROOT/p for p in model_paths)
    paths.extend([ROOT/'config/510300_post_selection_extension_inputs_v1.json',ROOT/'research/post_selection_extension_inputs_v1.py',ROOT/'reports/research/510300_round212_numeric_representation_fix.json',ROOT/'config/510300_post_selection_extension_inputs_v1_1.json',ROOT/'research/post_selection_extension_inputs_v1_1.py',ROOT/'reports/research/510300_round212_datetime_dtype_fix.json'])
    frozen=[{'path':str(p.relative_to(ROOT)), 'sha256':digest(p)} for p in sorted(set(paths))]
    OUT.mkdir(parents=True,exist_ok=True)
    rule=ROOT/'docs/510300_POST_SELECTION_EXTENSION_INPUTS_V1.md'
    require(rule.is_file(), '原冻结中文范围缺失')
    cfg={k:old_cfg[k] for k in ['evaluation_start','data_cutoff','initial_capital','annual_days','cash_annual_rate_assumption',
        'high_sharpe_target','annual_return_target','costs','features','dividends','earlier_start','earlier_terminal']}
    cfg.update(study_id='510300_POST_SELECTION_EXTENSION_INPUTS_V1', round=212, registered_at=now(), primary=PRIMARY,
        candidate_models=[], candidate_configurations=0, rules=str(rule.relative_to(ROOT)), frozen_files=frozen,
        network_requests=0,new_model_fits=0,new_reference_accounts=0,goal_achieved=False,
        evidence_class='PRE_SELECTION_HISTORICAL_GAP_ENGINEERING_INPUTS',admission_source='SINA_UNROUNDED_CONTINUATION',
        volume_rounding_tolerance_shares=50,amount_rounding_tolerance_cny=50,complete_checkpoint_verified=False,
        numerical_representation_tolerance_cny=1e-6,amends='config/510300_post_selection_extension_inputs_v1_1.json',
        correction_scope='仅将腾讯万元浮点换算到元的不足一微元表示误差与行情百单位舍入差异分开，不修改行情、因子或策略')
    write_json(CONFIG,cfg,exclusive=True)
    index['running_studies']=[{'round':212,'study':cfg['study_id'],'status':'INPUT_ADMISSION_RUNNING','config':str(CONFIG.relative_to(ROOT))}]
    write_json(index_path,index)
    old=pd.read_parquet(paths[0]);sina=pd.read_parquet(PREVIOUS/'sina_daily.parquet');tencent=pd.read_parquet(PREVIOUS/'tencent_daily.parquet')
    require(pd.DatetimeIndex(sina.date).equals(pd.DatetimeIndex(tencent.date)), '两来源日期不一致')
    rows=[]
    for field in ['open','high','low','close','volume','amount']:
        a,b=sina[field].to_numpy(float),tencent[field].to_numpy(float)
        require(np.isfinite(a).all() and np.isfinite(b).all(), '来源字段存在缺失')
        differences=np.abs(a-b)
        threshold=50 if field in ['volume','amount'] else 0
        require((differences<=threshold).all(), '两来源存在超出精度范围的差异：'+field)
        if threshold:
            require(np.array_equal(np.round(a/100)*100,np.rint(b)) and np.all(np.abs(b-np.rint(b))<=1e-6), '腾讯尾数未能在一微元数值表示精度内由新浪按百单位舍入复现：'+field)
        for date,x,y,d in zip(sina.date,a,b,differences):
            rows.append(dict(date=date,field=field,sina=x,tencent=y,absolute_difference=d,tolerance=threshold))
    require((OUT/'source_precision_comparison.csv').is_file(), '已保存精度比较缺失')
    saved_comparison=pd.read_csv(OUT/'source_precision_comparison.csv',parse_dates=['date'])
    pd.testing.assert_frame_equal(saved_comparison,pd.DataFrame(rows),check_dtype=False,rtol=0,atol=1e-6)
    overlap=old.merge(sina,on='date',suffixes=('_old','_new'))
    require(len(overlap)==10,'旧新区间重叠日数不同')
    for field in ['open','high','low','close','volume','amount']:
        require(np.array_equal(overlap[field+'_old'],overlap[field+'_new']), '新浪旧重叠区间被修订：'+field)
    extra=sina[sina.date>old.date.max()].copy()
    receipt=read(PREVIOUS/'sina_receipt.json')
    for column in old.columns:
        if column not in extra:
            extra[column]=old[column].iloc[-1]
    extra['source']='sina.klc_saved_211';extra['source_original']='sina.klc_saved_211'
    extra['retrieved_at']=receipt['retrieved_at'];extra['correction_applied']=False;extra['correction_reason']=''
    extra=extra[old.columns].astype(old.dtypes.to_dict())
    candidate=pd.concat([old,extra],ignore_index=True)
    expected=read(PREVIOUS/'time_boundaries.json')['expected_gap_dates_before_strategy_freeze']
    require([str(d.date()) for d in extra.date]==expected and len(candidate)==3476, '补充日期或总行数不同')
    pd.testing.assert_frame_equal(candidate.iloc[:len(old)].reset_index(drop=True),old,check_exact=True)
    candidate.to_parquet(OUT/'candidate_prices.parquet',index=False)
    feature,feature_columns=factors(candidate,pd.read_csv(ROOT/cfg['dividends']))
    saved=pd.read_parquet(ROOT/cfg['features'])
    require(list(feature.columns)==list(saved.columns), '特征列结构不同')
    numeric=saved.select_dtypes(include=['number','bool']).columns
    comparisons=[]
    for column in numeric:
        a=saved[column].to_numpy(float);b=feature[column].iloc[:len(saved)].to_numpy(float)
        exact=bool(np.array_equal(a,b,equal_nan=True))
        require(exact,'原特征前缀改变：'+column)
        comparisons.append(dict(column=column,rows=len(saved),exact_equal=True,maximum_absolute_difference=0,
            missing_values=int(np.isnan(a).sum())))
    require(feature.feature_valid.iloc[-20:].all(), '新增日期特征有不完整行')
    feature.to_parquet(OUT/'candidate_features.parquet',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'old_feature_prefix_comparison.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'dependency_graph.json',{'root':PRIMARY,'nodes':nodes,'topological_order':[n['node'] for n in nodes],
        'required_account_replays_main_only':len(inventory),'shared_nodes_computed_once':True,
        'excluded':'零权重外层来源、未选批次设置、纯目标层的无用账户、109条件方差拟合及其父账户、旧控制账户'},exclusive=True)
    pd.DataFrame([{**n,'direct_parents':'；'.join(n['direct_parents'])} for n in nodes]).to_csv(OUT/'minimal_dependencies.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(inventory).to_csv(OUT/'required_account_state_inventory.csv',index=False,encoding='utf-8-sig')
    model_info=[]
    for path in model_paths:
        records=read(path)['models'];records=records['D60_INTRA__RIDGE'] if isinstance(records,dict) else records
        model_info.append(dict(path=path,sha256=digest(ROOT/path),records=len(records),last_fit_origin=records[-1]['fit_origin'],
            last_fit_time=records[-1]['fit_time'],september_model_present=any(r['fit_origin']=='2026-09-01' for r in records),
            adaptation='先验证旧区间模型逐时复用与状态恢复；九月月首新增训练须仅按原日程与已成熟周期生成，不能把沿用八月模型称为已完成月度更新'))
    write_json(OUT/'model_tail_inventory.json',{'sources':model_info,'new_model_fits':0},exclusive=True)
    coverage=read(PREVIOUS/'dividend_candidate_coverage.json')
    require(coverage['status']=='OFFICIAL_CANDIDATE_COVERAGE_CONFIRMED_NO_LEDGER_CHANGE','候选分红覆盖未通过')
    write_json(OUT/'candidate_dividend_coverage.json',{'source':str((PREVIOUS/'dividend_candidate_coverage.json').relative_to(ROOT)),
        'coverage':coverage,'ledger':cfg['dividends'],'ledger_sha256':digest(ROOT/cfg['dividends']),'global_coverage_overwritten':False},exclusive=True)
    original=read('reports/research/510300_incremental_selected_intent_mix_v1/result.json')
    result=dict(study_id=cfg['study_id'],completed_at=now(),status='COMPLETED_INPUT_PREFIX_AND_MINIMAL_DEPENDENCIES',
        candidate_configurations=0,candidate_models=[],evaluation_accounts=0,new_accounts_generated=0,reused_control_accounts=0,
        earlier_diagnostic_accounts=0,new_earlier_diagnostic_accounts=0,new_model_fits=0,new_reference_accounts=0,
        all_metrics=[r for r in original['all_metrics'] if r['model']==PRIMARY],
        earlier_diagnostics=[r for r in original['earlier_diagnostics'] if r['model']==PRIMARY],metric_context_only_reused_from_round209=True,
        primary=PRIMARY,post_selected_best_base=next(r for r in original['all_metrics'] if r['model']==PRIMARY and r['cost']=='BASE'),
        goal_achieved=False,independent_validation='NOT_ESTABLISHED',position_impact=0,
        old_price_rows=len(old),new_price_rows=len(extra),candidate_price_rows=len(candidate),old_numeric_feature_columns=len(numeric),
        old_numeric_cells_compared=len(old)*len(numeric),old_feature_prefix_exact=True,new_feature_valid_rows=20,
        comparison_rows=len(rows),maximum_volume_difference=max(r['absolute_difference'] for r in rows if r['field']=='volume'),
        maximum_amount_difference=max(r['absolute_difference'] for r in rows if r['field']=='amount'),
        dependency_nodes=len(nodes),required_account_replays_main_only=len(inventory),old_model_records=sum(x['records'] for x in model_info),
        candidate_inputs_ready_for_engineering=True,data_accepted_into_strategy=False,new_strategy_performance_computed=False,
        strict_forward_evidence_days=0,network_requests=0,source_cost_cny=0,run_seconds=time.perf_counter()-began)
    write_json(OUT/'result.json',result,exclusive=True)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/'candidate_prices.parquet'),candidate,check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/'candidate_features.parquet'),feature,check_exact=True)
    for item in frozen:
        require(digest(ROOT/item['path'])==item['sha256'],'原依赖或输入发生改变')
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':0,
        'scope':'本项为输入和依赖接纳，直接核对保存价格、完整旧因子前缀、舍入及依赖，不新增策略测试'},exclusive=True)
    write_json(OUT/'saved_verification_receipt.json',{'verified_at':now(),'status':'PASS_SAVED_INPUTS_EXACT_PREFIX_AND_REACHABLE_DEPENDENCIES',
        'old_inputs_unchanged':True,'numeric_cells_compared':result['old_numeric_cells_compared'],'dependency_nodes':len(nodes),
        'required_source_account_files_verified':len(inventory),'new_accounts':0,'new_model_fits':0,
        'complete_recursive_checkpoint_verified':False,'independent_performance_validation':False},exclusive=True)
    print(json.dumps({k:result[k] for k in ['status','candidate_price_rows','old_numeric_cells_compared','dependency_nodes',
        'required_account_replays_main_only','maximum_volume_difference','maximum_amount_difference','run_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
