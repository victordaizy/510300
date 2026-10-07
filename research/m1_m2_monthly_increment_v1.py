"""冻结的货币剪刀差月频增量研究；数据、预测与账户准入分离。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from research.m1_m2_release_sources_v1 import ROOT, OUT, RAW, now, sha, write_json

CONFIG = ROOT / 'config/510300_m1_m2_monthly_increment_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv(path, rows):
    if path.exists():
        raise FileExistsError('禁止覆盖已有研究产物：' + str(path))
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def source_admission():
    cfg = read(CONFIG)
    parts = [pd.read_csv(OUT / name) for name in ['official_release_candidates.csv',
             'supplemental_release_candidates.csv', 'january_2025_release_candidates.csv']]
    candidates = pd.concat(parts, ignore_index=True)
    good = candidates[candidates.admission.eq('DATED_REPORT_PARSED_PENDING_DEFINITION_ADMISSION')].copy()
    assert good.stat_month.is_unique
    expected = pd.period_range(cfg['source_start_month'], cfg['source_end_month'], freq='M').astype(str)
    assert set(good.stat_month) == set(expected), '仍有月份缺少当次官方报告'
    good = good.sort_values('stat_month').reset_index(drop=True)
    versions = cfg['methodology_adjudication']['definition_versions']
    rows = []
    for item in good.to_dict('records'):
        month = item['stat_month']
        version = next(v for v in versions if v['start'] <= month <= v['end'])
        published = pd.Timestamp(item['published_at'])
        assert published.tzinfo is not None
        assert pd.Period(month, freq='M').end_time.date() < published.date()
        raw = ROOT / item['raw_path']
        assert sha(raw) == item['source_sha256']
        item.update(definition_version=version['version'], training_regime=version['training_regime'],
                    spread_pp=item['m1_yoy_pp'] - item['m2_yoy_pp'],
                    admission='ADMITTED_RECONSTRUCTED_OFFICIAL_RELEASE',
                    available_before_market_cutoff=published <= pd.Timestamp(cfg['data_cutoff'] + 'T15:00:00+08:00'))
        rows.append(item)
    releases = pd.DataFrame(rows)
    csv(OUT / 'admitted_releases.csv', releases)
    features = []
    lookup = releases.set_index('stat_month')
    for row in rows:
        month = pd.Period(row['stat_month'], freq='M')
        lag = str(month - cfg['macro_lag_months'])
        current = {k: row[k] for k in ['stat_month', 'published_at', 'available_at_upper_bound',
                   'definition_version', 'training_regime', 'source_url', 'source_sha256',
                   'm1_yoy_pp', 'm2_yoy_pp', 'spread_pp', 'available_before_market_cutoff']}
        current.update(lag_month=lag, delta3_spread_pp=np.nan, delta3_m1_pp=np.nan, delta3_m2_pp=np.nan,
                       lag_published_at=None, lag_source_sha256=None, lag_definition_version=None)
        if lag not in lookup.index:
            current['feature_status'] = 'NO_VIEW_MISSING_THREE_MONTH_LAG'
        else:
            earlier = lookup.loc[lag]
            current.update(lag_published_at=earlier.published_at, lag_source_sha256=earlier.source_sha256,
                           lag_definition_version=earlier.definition_version)
            if earlier.definition_version != row['definition_version']:
                current['feature_status'] = 'NO_VIEW_DEFINITION_BREAK'
            else:
                assert pd.Timestamp(earlier.published_at) < pd.Timestamp(row['published_at'])
                d1 = row['m1_yoy_pp'] - earlier.m1_yoy_pp
                d2 = row['m2_yoy_pp'] - earlier.m2_yoy_pp
                current.update(delta3_spread_pp=d1 - d2, delta3_m1_pp=d1, delta3_m2_pp=d2,
                               feature_status='AVAILABLE',
                               component_case=('M1_UP_M2_DOWN' if d1 > 0 and d2 < 0 else
                                               'M1_UP_ONLY' if d1 > 0 else
                                               'M2_DOWN_ONLY' if d2 < 0 else 'NEITHER'))
        features.append(current)
    csv(OUT / 'release_features.csv', features)
    january = releases[releases.stat_month.eq('2025-01')].iloc[0]
    soup = BeautifulSoup((ROOT / january.raw_path).read_text(encoding='utf-8'), 'html.parser')
    tables = [t for t in soup.find_all('table') if '1120120' in t.get_text()]
    table = min(tables, key=lambda t: len(t.get_text()))
    cells = [[c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])] for tr in table.find_all('tr')]
    backcasts = []
    for group in [0, 3]:
        for col in range(1, 7):
            month_num = col + (0 if group == 0 else 6)
            value = float(cells[group + 2][col].replace('%', ''))
            old = lookup.loc[f'2024-{month_num:02d}']
            backcasts.append({'stat_month': f'2024-{month_num:02d}', 'm1_new_yoy_pp': value,
                             'm1_original_yoy_pp': old.m1_yoy_pp, 'revision_difference_pp': value-old.m1_yoy_pp,
                             'known_from': january.published_at, 'source_url': january.source_url,
                             'source_sha256': january.source_sha256,
                             'usable_as_2024_prediction_origin': False, 'used_in_primary_feature': False})
    csv(OUT / 'm1_2024_backcast_vintage.csv', backcasts)
    f = pd.DataFrame(features)
    write_json(OUT / 'source_admission.json', {
        'completed_at': now(), 'status': 'PASS_RECONSTRUCTED_RELEASES_WITH_EXPLICIT_VERSION_BOUNDARIES',
        'official_report_months': len(releases), 'first_stat_month': releases.stat_month.iloc[0],
        'last_stat_month': releases.stat_month.iloc[-1], 'original_attempt_failures_preserved': int((candidates.source_status != 'FETCHED').sum()),
        'publication_precision_counts': releases.publication_time_precision.value_counts().to_dict(),
        'before_market_cutoff': int(releases.available_before_market_cutoff.sum()),
        'feature_status_counts': f.feature_status.value_counts().to_dict(), 'backcast_rows': len(backcasts),
        'historical_immutable_vintages_proven': False, 'strict_forward_evidence_months': 0,
        'methodology_adjudication': cfg['methodology_adjudication'], 'market_data_reads': 0,
        'new_model_fits': 0, 'new_accounts': 0})
    print('当次公告、三个月差分及2024回溯版本已分开保存，尚未读取市场收益。', flush=True)


def ridge_fit(x, y, current, cfg):
    x, y, current = np.asarray(x, float), np.asarray(y, float), np.asarray(current, float)
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = np.clip((x-mean)/scale, -cfg['standardized_clip'], cfg['standardized_clip'])
    zmean = z.mean(axis=0)
    centered = z-zmean
    coef = np.linalg.solve(centered.T@centered + len(x)*cfg['ridge_mean_loss_penalty']*np.eye(x.shape[1]),
                           centered.T@(y-y.mean()))
    intercept = float(y.mean()-zmean@coef)
    current_z = np.clip((current-mean)/scale, -cfg['standardized_clip'], cfg['standardized_clip'])
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': coef.tolist(), 'intercept': intercept,
            'current_z': current_z.tolist(), 'prediction': float(intercept+current_z@coef)}


def make_monthly(data, releases, cfg):
    from research.monthly_single_factor_walkforward_v1 import monthly_samples
    from research.adaptive_allocation_v1 import normalize_dividends
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    samples = monthly_samples(data, dividends)
    release = releases.copy()
    release['available'] = pd.to_datetime(release.available_at_upper_bound, utc=True)
    seen, rows = set(), []
    for row in samples.to_dict('records'):
        if row['origin'] < pd.Timestamp(cfg['source_start_month']+'-01'):
            continue
        stamp = row['origin'].tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
        available = release[release.available.le(stamp.tz_convert('UTC'))]
        state = {'origin_clock': stamp.isoformat(), 'source_status': 'NO_VIEW_NO_RELEASE',
                 'stat_month': None, 'training_regime': None, 'definition_version': None,
                 'source_sha256': None, 'published_at': None, 'delta3_spread_pp': np.nan,
                 'delta3_m1_pp': np.nan, 'delta3_m2_pp': np.nan, 'component_case': None,
                 'release_age_days': np.nan}
        if len(available):
            latest = available.sort_values('available').iloc[-1]
            state.update({k: latest.get(k) for k in ['stat_month','training_regime','definition_version',
                         'source_sha256','published_at','delta3_spread_pp','delta3_m1_pp','delta3_m2_pp','component_case']})
            age = (stamp.tz_convert('UTC')-latest.available).total_seconds()/86400
            status = latest.feature_status
            if age > cfg['source_contract']['publication_age_limit_days']:
                status = 'NO_VIEW_STALE_RELEASE'
            if latest.stat_month in seen:
                status = 'NO_VIEW_REPEATED_RELEASE'
            seen.add(latest.stat_month)
            state.update(source_status=status, release_age_days=age)
        rows.append({**row, **state})
    return pd.DataFrame(rows)


def bootstrap_indices(n, repetitions, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n,size=(repetitions,math.ceil(n/block)))
    return ((starts[:,:,None]+np.arange(block)) % n).reshape(repetitions,-1)[:,:n]


def assess(frame, cfg, label):
    paired = frame[frame.fit_status.eq('FIT_READY') & frame.label.notna()].copy().reset_index(drop=True)
    n = len(paired)
    if not n:
        return {'regime': label, 'status': 'NO_VIEW_INSUFFICIENT_MATURE_TRAINING_MONTHS',
                'paired_months': 0, 'pass': False, 'gates': []}, None, None
    y = paired.label.to_numpy(float)
    loss0 = (y-paired.baseline_prediction.to_numpy(float))**2
    loss1 = (y-paired.increment_prediction.to_numpy(float))**2
    gain = loss0-loss1
    state = paired.delta3_spread_pp.to_numpy(float)>0
    ix = bootstrap_indices(n,cfg['bootstrap_repetitions'],cfg['bootstrap_block_months'],cfg['random_seed'])
    draws = gain[ix].mean(axis=1)
    state_draws = state[ix]
    counts = state_draws.sum(axis=1)
    group_draws = np.divide((y[ix]*state_draws).sum(axis=1),counts,out=np.full(len(ix),np.nan),where=counts>0)-np.divide(
        (y[ix]*(~state_draws)).sum(axis=1),n-counts,out=np.full(len(ix),np.nan),where=counts<n)
    spread = float(y[state].mean()-y[~state].mean()) if state.any() and (~state).any() else None
    lower = float(np.quantile(draws,cfg['bootstrap_lower_quantile']))
    group_lower = float(np.nanquantile(group_draws,cfg['bootstrap_lower_quantile']))
    halves = [float(gain[:n//2].mean()),float(gain[n//2:].mean())]
    years = pd.to_datetime(paired.origin).dt.year.to_numpy()
    loo = {str(year):float(gain[years!=year].mean()) for year in sorted(set(years)) if np.any(years!=year)}
    gates = [
        {'gate':'minimum_paired_months','pass':n>=cfg['minimum_paired_evaluation_months'],'value':n},
        {'gate':'minimum_state_months','pass':min(int(state.sum()),int((~state).sum()))>=cfg['minimum_months_per_improvement_state'],'value':[int(state.sum()),int((~state).sum())]},
        {'gate':'positive_mse_increment_and_lower','pass':float(gain.mean())>0 and lower>0,'value':float(gain.mean()),'lower_5pct':lower},
        {'gate':'both_halves_positive','pass':all(v>0 for v in halves),'value':halves},
        {'gate':'positive_state_spread_and_lower','pass':spread is not None and spread>0 and group_lower>0,'value':spread,'lower_5pct':group_lower},
        {'gate':'all_leave_one_year_out_positive','pass':bool(loo) and all(v>0 for v in loo.values()),'value':loo}]
    result = {'regime':label,'paired_months':n,'first_origin':str(paired.origin.iloc[0]),'last_origin':str(paired.origin.iloc[-1]),
              'mse_baseline':float(loss0.mean()),'mse_increment':float(loss1.mean()),'mse_gain':float(gain.mean()),
              'relative_mse_reduction':float(gain.mean()/loss0.mean()),'mse_gain_lower_5pct':lower,
              'improving_state_months':int(state.sum()),'non_improving_state_months':int((~state).sum()),
              'mean_return_improving':float(y[state].mean()),'mean_return_not_improving':float(y[~state].mean()),
              'conditional_return_spread':spread,'conditional_return_spread_lower_5pct':group_lower,
              'coefficient_positive_fraction':float((paired.macro_coefficient>0).mean()),
              'gates':gates,'pass':all(g['pass'] for g in gates),
              'status':'PASS_HISTORICAL_INCREMENT_ONLY' if all(g['pass'] for g in gates) else 'REJECTED_FIXED_INCREMENT_GATES'}
    paired['baseline_loss'],paired['increment_loss'],paired['loss_gain'] = loss0,loss1,gain
    csv(OUT / f'paired_{label}.csv',paired)
    csv(OUT / f'bootstrap_{label}.csv',{'mse_gain':draws,'conditional_return_spread':group_draws})
    np.savez_compressed(OUT / f'bootstrap_indices_{label}.npz',indices=ix)
    annual = paired.assign(year=years).groupby('year').agg(months=('label','size'),mse_gain=('loss_gain','mean'),
                          mean_return=('label','mean'),delta3_mean=('delta3_spread_pp','mean')).reset_index()
    csv(OUT / f'annual_{label}.csv',annual)
    components = paired.groupby('component_case').agg(months=('label','size'),mean_return=('label','mean'),
                            median_return=('label','median'),mean_delta_m1=('delta3_m1_pp','mean'),
                            mean_delta_m2=('delta3_m2_pp','mean')).reset_index()
    csv(OUT / f'components_{label}.csv',components)
    return result, paired, ix


def freeze():
    cfg=read(CONFIG)
    assert read(OUT/'source_admission.json')['market_data_reads']==0
    tests=read(OUT/'tests_receipt.json')
    assert tests['passed'] and tests['exit_code']==0
    paths=[CONFIG,Path(__file__),ROOT/'research/m1_m2_release_sources_v1.py',ROOT/'docs/510300_M1_M2_MONTHLY_INCREMENT_V1.md',
           ROOT/'tests/test_m1_m2_monthly_increment_v1.py',OUT/'tests_receipt.json',OUT/'source_admission.json',
           OUT/'admitted_releases.csv',OUT/'release_features.csv',OUT/'m1_2024_backcast_vintage.csv',
           ROOT/cfg['market_features'],ROOT/cfg['dividends'],ROOT/'research/monthly_single_factor_walkforward_v1.py',
           ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'config/510300_research_authority_v6.json',OUT/'USER_REQUEST.md']
    paths += sorted(RAW.glob('*'))
    protocol={**cfg,'registered_at':now(),'macro_prediction_results_read':False,
              'source_reconstruction_completed_before_registration':True,
              'historical_price_and_old_study_results_previously_seen':True,
              'frozen_files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]}
    write_json(OUT/'protocol.json',protocol)
    print('宏观增量协议及直接来源已冻结；下面只允许本固定方法的一次运行。',flush=True)


def check_protocol():
    cfg=read(OUT/'protocol.json')
    for item in cfg['frozen_files']:
        assert sha(ROOT/item['path'])==item['sha256'], '冻结文件变化：'+item['path']
    return cfg


def predict():
    cfg=check_protocol()
    write_json(OUT/'PREDICTION_STARTED.json',{'started_at':now(),'protocol_sha256':sha(OUT/'protocol.json'),
                                           'parameter_search':False})
    data=pd.read_parquet(ROOT/cfg['market_features'])
    data['date']=pd.to_datetime(data.date)
    assert data.date.is_monotonic_increasing and data.date.is_unique
    assert data.date.iloc[-1]==pd.Timestamp(cfg['data_cutoff'])
    data['logvol20']=np.log(data.vol20.where(data.vol20>0))
    release=pd.read_csv(OUT/'release_features.csv')
    samples=make_monthly(data,release,cfg)
    csv(OUT/'monthly_samples.csv',samples)
    pcfg=cfg['prediction']
    features=pcfg['increment_features']
    valid=samples.source_status.eq('AVAILABLE') & np.isfinite(samples[features+['label']].to_numpy(float)).all(axis=1)
    forecasts,members,models=[],[],[]
    for row in samples.to_dict('records'):
        train=samples[valid & samples.training_regime.eq(row['training_regime']) & samples.mature_date.le(row['origin'])].tail(pcfg['training_months'])
        supported=row['source_status']=='AVAILABLE' and len(train)>=pcfg['minimum_training_months']
        supported=supported and bool(np.isfinite([row[f] for f in features]).all())
        item={**row,'training_count':len(train),'training_last_maturity':train.mature_date.max(),
              'fit_status':'FIT_READY' if supported else ('NO_VIEW_INSUFFICIENT_MATURE_TRAINING_MONTHS' if row['source_status']=='AVAILABLE' else row['source_status']),
              'baseline_prediction':np.nan,'increment_prediction':np.nan,'macro_coefficient':np.nan}
        if supported:
            assert train.origin.lt(row['origin']).all() and train.mature_date.le(row['origin']).all()
            for name,fs in [('baseline',pcfg['baseline_features']),('increment',features)]:
                fit=ridge_fit(train[fs],train.label,[row[f] for f in fs],pcfg)
                item[name+'_prediction']=fit['prediction']
                if name=='increment':
                    item['macro_coefficient']=fit['coef'][-1]
                models.append({'origin':str(row['origin']),'model':name,'features':fs,'training_regime':row['training_regime'],
                               'training_origins':train.origin.astype(str).tolist(),**fit})
            members.extend({'fit_origin':row['origin'],'sample_origin':z.origin,'mature_date':z.mature_date,
                            'regime':z.training_regime,'source_sha256':z.source_sha256} for z in train.itertuples())
        forecasts.append(item)
    forecasts=pd.DataFrame(forecasts)
    csv(OUT/'monthly_forecasts.csv',forecasts)
    csv(OUT/'training_members.csv',members)
    write_json(OUT/'saved_models.json',models)
    outcomes=[]
    for regime in sorted(release.training_regime.unique()):
        result,_,_=assess(forecasts[forecasts.training_regime.eq(regime)],pcfg,regime)
        outcomes.append(result)
    historical=next(r for r in outcomes if r['regime']=='M1_OLD_M2_MMF2018')
    result={'study_id':cfg['study_id'],'completed_at':now(),
            'status':'HISTORICAL_INCREMENT_PASS_CURRENT_REGIME_UNVALIDATED' if historical['pass'] else 'REJECTED_FROZEN_OLD_DEFINITION_NO_RELIABLE_INCREMENT_NEW_DEFINITION_INSUFFICIENT',
            'historical_macro_gate_pass':historical['pass'],'current_definition_gate_pass':False,
            'outcomes':outcomes,'source_status_counts':forecasts.source_status.value_counts().to_dict(),
            'fit_status_counts':forecasts.fit_status.value_counts().to_dict(), 'new_model_fits':len(models),
            'new_candidate_macro_features':1,'new_risk_candidate_methods':0,'new_accounts':0,
            'B_account_stage':'ELIGIBLE_OLD_DEFINITION_REQUIRES_ACCOUNT_REGISTRATION' if historical['pass'] else 'NOT_RUN_PREDICTION_GATE',
            'D_account_stage':'ELIGIBLE_OLD_DEFINITION_REQUIRES_ACCOUNT_REGISTRATION' if historical['pass'] else 'NOT_RUN_PREDICTION_GATE',
            'B_CAGR':None,'B_Sharpe':None,'D_CAGR':None,'D_Sharpe':None,
            'strict_forward_evidence_months':0,'independent_validation':False,
            'DSR':'NOT_COMPUTED_MISSING_EFFECTIVE_TRIAL_UNIVERSE','performance_goal_achieved':False,'position_impact':0}
    write_json(OUT/'prediction_result.json',result)
    print(json.dumps({'状态':result['status'],'拟合数':len(models),'各口径':outcomes},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='一次性M1/M2月频增量研究')
    parser.add_argument('stage',choices=['admit','freeze','predict'])
    args=parser.parse_args()
    {'admit':source_admission,'freeze':freeze,'predict':predict}[args.stage]()
