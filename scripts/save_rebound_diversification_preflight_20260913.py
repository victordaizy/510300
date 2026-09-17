"""仅保存两条既有主历史账本的相关性及较早来源缺口，不计算新策略。"""
import json
from pathlib import Path

import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/research/510300_saved_rebound_diversification_preflight_20260913'
    require(not (out/'result.json').exists(),'这项保存来源诊断已经完成')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/510300_account_volatility_exposure_v1.json').read_text(encoding='utf-8'))
    rows,sources,missing=[],[],[]
    for cost in ['BASE','STRESS']:
        core=ROOT/f'reports/research/510300_account_volatility_exposure_v1/evaluation/{cost}/ACCOUNT_VOLATILITY_EXPOSURE_ledger.parquet'
        aux=ROOT/f'reports/research/510300_simple_price_entry_exit_v1/evaluation/{cost}/R2_Z_CONFIRM_ledger.parquet'
        a,b=pd.read_parquet(core),pd.read_parquet(aux)
        require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)),'已有两账户日历不同')
        m=summarize(b,cfg)
        rows.append({'period':'evaluation','cost':cost,'days':len(a),'pearson_net_return_correlation':float(a.net_return.corr(b.net_return)),
            'auxiliary_sharpe':m['net_sharpe'],'auxiliary_annual_return':m['annualized_return'],'auxiliary_trades':m['trade_count']})
        sources.extend({'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in [core,aux])
        old=ROOT/f'reports/research/510300_simple_price_entry_exit_v1/earlier_diagnostic/{cost}/R2_Z_CONFIRM_ledger.parquet'
        require(not old.exists(),'较早来源状态已改变，请读已有记录')
        missing.append(str(old.relative_to(ROOT)))
    write_json(out/'result.json',{'checked_at':now(),'status':'SAVED_MAIN_LEDGER_CORRELATION_AND_EARLIER_SOURCE_GAP',
        'rows':rows,'sources':sources,'missing_earlier_source_paths':missing,'new_accounts':0,'new_model_fits':0,
        'new_strategy_return_read':False,'interpretation':'主历史低相关仅用于选择待检验组合，不代表有可盈利增量；较早来源须另行明确登记两条参考账户。'},exclusive=True)
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
