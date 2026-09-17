import json
from research.core_auxiliary_drawdown_gate_v1 import ROOT,OUT,CONFIG,PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,write_json
def main():
 r=json.loads((OUT/'result.json').read_text(encoding='utf-8'));a=[x for x in r['all_metrics'] if x['model']==PRIMARY];e=[x for x in r['earlier_diagnostics'] if x['model']==PRIMARY]
 decision='第176轮六十日市场回撤在小于或等于负5%时撤去辅助仓位。主历史基础／压力夏普为1.608／1.534，较早为0.855／0.832。主历史改善但较早历史显著低于1.2，四场景稳定目标未完成，关闭本固定阈值方案。'
 detail='三项必要测试通过，四个完整账户核心计算2.39秒。独立重建5646个目标并核对全部账户决定、106段实际持仓、费用、分红和终点清算。主历史年化超额为4.055／3.698个百分点，较早为2.208／2.102个百分点；较早最大回撤9.82%／10.54%，高于此前任一方向确认的约6.30%。市场回撤条件在两个历史段表现方向相反，不能以主历史改善宣布达标。'
 write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,PRIMARY:'CLOSED_MAIN_IMPROVED_EARLIER_SHARPE_BELOW12'},exclusive=True);n=ROOT/'docs/510300_DRAWDOWN_GATE_RECOVERY_NEXT_20260912.md';d=ROOT/'deliverables/510300市场回撤辅助门_第176轮_20260912/市场回撤辅助门_结果及全部中文规则.md';deliver_round(ROOT,OUT,CONFIG,d,'六十日市场回撤撤去辅助仓位','CLOSED_CORE_AUXILIARY_DRAWDOWN_GATE_FULL_GOAL_NOT_MET',decision,detail,n,n.read_text(encoding='utf-8').splitlines(),'DRAWDOWN_GATE_RECOVERY_IDEA_PREPARED','市场回撤解除后的辅助恢复确认')
 p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json';i=json.loads(p.read_text(encoding='utf-8'));i['next_work'].update(candidate_round=177,registered=False,planned_settings=0,planned_new_accounts=0);i['current_best_four_scenario_comparison_candidate']['last_compared_completed_round']=176;i['latest_main_sharpe_pass_candidate']['last_compared_completed_round']=176;i['latest_all_four_scenario_excess_candidate']['last_compared_completed_round']=176;write_json(p,i);print('第176轮交付完成。')
if __name__=='__main__':main()
