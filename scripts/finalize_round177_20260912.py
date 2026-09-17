import json
from research.drawdown_gate_recovery_v1 import ROOT,OUT,CONFIG,PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,write_json
def main():
 r=json.loads((OUT/'result.json').read_text(encoding='utf-8'));decision='第177轮要求市场回撤恢复后连续两个收盘仍安全，才重新加入辅助。主历史夏普仍为1.608／1.534，较早仅0.867／0.848。较早历史依旧远低于1.2，关闭本恢复确认方案。';detail='两项必要测试通过，四个完整账户核心计算1.53秒。独立核对5646个账户决定和102段实际持仓。与第176轮相比，主历史所有主要指标完全相同；较早夏普仅由0.855／0.832升至0.867／0.848，仍显著不足，不能把延迟恢复视为有效改进。'
 write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,PRIMARY:'CLOSED_MAIN_UNCHANGED_EARLIER_BELOW12'},exclusive=True);n=ROOT/'docs/510300_AUXILIARY_RECOVERY_MOMENTUM_NEXT_20260912.md';d=ROOT/'deliverables/510300回撤恢复确认_第177轮_20260912/回撤恢复确认_结果及全部中文规则.md';deliver_round(ROOT,OUT,CONFIG,d,'六十日市场回撤两日恢复后加入辅助','CLOSED_DRAWDOWN_GATE_RECOVERY_FULL_GOAL_NOT_MET',decision,detail,n,n.read_text(encoding='utf-8').splitlines(),'AUXILIARY_RECOVERY_MOMENTUM_IDEA_PREPARED','回撤恢复与累计方向共同决定辅助仓位')
 p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json';i=json.loads(p.read_text(encoding='utf-8'));i['next_work'].update(candidate_round=178,registered=False,planned_settings=0,planned_new_accounts=0);i['current_best_four_scenario_comparison_candidate']['last_compared_completed_round']=177;i['latest_main_sharpe_pass_candidate']['last_compared_completed_round']=177;i['latest_all_four_scenario_excess_candidate']['last_compared_completed_round']=177;write_json(p,i);print('第177轮交付完成。')
if __name__=='__main__':main()
