"""保持第26轮恐慌回升规则不变，只检验更早五年历史。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research.simple_volume_reversal_v1 import make_rules
from research.simple_price_entry_exit_v1 import simulate_policy
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import normalize_dividends,save_account,summarize
from research.intraday_overnight_increment_v1 import digest,now,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_panic_earlier_history_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/510300_simple_volume_reversal_v1.json').read_text(encoding='utf-8'))
    write_json(OUT/'protocol.json',{'registered_at':now(),'purpose':'原参数检验2015至2019，不参与2020以后策略选择，不替代既定全时期目标',
               'start':'2015-01-05','terminal':'2019-12-31','source_config_sha256':digest(ROOT/'config/510300_simple_volume_reversal_v1.json'),
               'candidate':'V6_PANIC_RECOVERY','parameters_changed':False,'position_impact':0},exclusive=True)
    data=pd.read_parquet(ROOT/cfg['features']);data=data[data.date<='2019-12-31'].copy()
    div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    rule=make_rules(data)[0]['V6_PANIC_RECOVERY'];results=[]
    for cost_id,cost in cfg['costs'].items():
        ledger,decisions,cycles=simulate_policy(data,div,cfg,cost,'2015-01-05',rule,cfg['candidate_specs']['V6_PANIC_RECOVERY'])
        save_account(OUT/cost_id,'V6_PANIC_RECOVERY',ledger,decisions)
        cycles.to_csv(OUT/cost_id/'cycles.csv',index=False,encoding='utf-8-sig')
        baseline,bd=simulate_event_account(data,div,cfg,cost,'2015-01-05','BUY_HOLD',event_mask=np.ones(len(data),bool))
        save_account(OUT/cost_id,'BUY_HOLD',baseline,bd)
        for key,frame in [('V6_PANIC_RECOVERY',ledger),('BUY_HOLD',baseline)]:
            results.append({'cost':cost_id,'model':key,**summarize(frame,cfg)})
    write_json(OUT/'result.json',{'completed_at':now(),'status':'UNCHANGED_RULE_EARLIER_HISTORY_DIAGNOSTIC_COMPLETE','metrics':results,
               'new_diagnostic_accounts':4,'new_candidate_configurations':0,'independent_validation':'NOT_ESTABLISHED_PREVIOUSLY_OBSERVED_EARLIER_HISTORY','goal_achieved':False},exclusive=True)
    print(json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
