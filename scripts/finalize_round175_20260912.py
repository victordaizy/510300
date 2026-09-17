"""关闭第175轮，交付中文结果并登记下一项未实现研究。"""
import json
from research.confirmed_auxiliary_episode_batch_v1 import ROOT,OUT,CONFIG,PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,write_json
def main():
 result=json.loads((OUT/'result.json').read_text(encoding='utf-8')); main=[x for x in result['all_metrics'] if x['model']==PRIMARY]; early=[x for x in result['earlier_diagnostics'] if x['model']==PRIMARY]
 decision='第175轮把方向确认改为辅助段资格。段首确认的主历史基础／压力夏普为1.483／1.417，较早为1.098／1.089；段内等待分别为1.480／1.407，较早同为1.098／1.089。四场景年化超额为正，但较早历史仍不足1.2，完整目标未完成。'
 detail='五项必要测试4.31秒通过；两套方案八个新账户核心计算16.59秒。独立重建11292个来源目标，并核对11292个账户决定、180段实际持仓及账户资金、费用、分红和终点清算。段内等待仅在主历史多取得九个资格，却未改善基础或压力夏普；较早历史两方案完全相同，说明该样本中辅助正目标段首已经覆盖了后来可等到的允许方向。\n\n主历史段首确认的年化收益为6.96%／6.63%，段内等待为6.98%／6.62%；较早历史两方案均为8.13%／8.11%。两者均不改变较早夏普不足的结论。未按结果再改资格规则、方向窗口、费用或历史起点。\n\n下一项仅准备核心与辅助的回撤上限分层，尚未登记、实现或运行。'
 write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,PRIMARY:'CLOSED_EARLIER_SHARPE_BELOW12_FOUR_EXCESSES_POSITIVE', 'EPISODE_WAIT_CONFIRMED_AUXILIARY':'CLOSED_EARLIER_SHARPE_BELOW12_FOUR_EXCESSES_POSITIVE'},exclusive=True)
 nxt=ROOT/'docs/510300_CORE_AUXILIARY_DRAWNDOWN_GATE_NEXT_20260912.md'; doc=ROOT/'deliverables/510300辅助资格段_第175轮_20260912/辅助资格段_结果及全部中文规则.md'
 deliver_round(ROOT,OUT,CONFIG,doc,'方向确认辅助资格段的开始与等候比较','CLOSED_CONFIRMED_AUXILIARY_EPISODE_BATCH_FULL_GOAL_NOT_MET',decision,detail,nxt,nxt.read_text(encoding='utf-8').splitlines(),'CORE_AUXILIARY_DRAWDOWN_GATE_IDEA_PREPARED','核心与辅助的账户回撤上限分层')
 p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'; index=json.loads(p.read_text(encoding='utf-8')); index['next_work'].update(candidate_round=176,registered=False,planned_settings=0,planned_new_accounts=0,planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False); index['current_best_four_scenario_comparison_candidate']['last_compared_completed_round']=175; index['latest_main_sharpe_pass_candidate']['last_compared_completed_round']=175; index['latest_all_four_scenario_excess_candidate']['last_compared_completed_round']=175;write_json(p,index);print('第175轮交付完成。')
if __name__=='__main__':main()
