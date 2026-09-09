"""交付长期趋势进入资格的跨阶段不同效果，结束该固定条件。"""
import json
from research.episode_trend_admission_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第147轮较早夏普提高，但主历史明显恶化，结束本长期趋势进入资格，不替换143。主基础／压力夏普0.676／0.640，"
        "年化1.76%／1.66%，最大回撤2.68%／2.82%；较早夏普1.183／1.215，年化4.69%／4.97%，最大回撤3.79%／3.88%。"
        "较早压力费用一项夏普超过1.2，较早基础仍低于1.2；四情景最低只有0.640。不能用一个阶段或一档费用达标代替完整目标。")
    detail = ("五项必要测试3.18秒，四账户核心1.063929秒，零模型拟合、新参考和下载；核心时间不含开发、测试、核对和文档。"
        "直接按父明确零分段、逐窗口均值还原200日趋势，核对5646个判断、四账户38个实际完整周期。"
        "主每档143原有23段正信号，接受12段、拒绝11段；其中22段已被父明确零结束，最后一段仍正但实际账户已终点开盘清仓。"
        "主各12实际周期、172持仓收盘37成交。较早每档10段接受7段、拒绝3段，全部信号段结束，各7实际周期、174／180持仓收盘24成交。"
        "所有资格明确，无未知目标或受阻请求，终点全部清仓。实际平均股票敞口主3.41%，较早6.41%／6.79%。"
        "相对143，主基础价格及分红毛收益少44124.90元，省佣金和滑点2250.28元，净少41874.62元；主压力净少39666.53元。"
        "较早基础毛收益少7663.50元、省费用842.38元、净少6821.12元；压力净少5976.03元。较早夏普提高主要体现在波动下降更多，年化收益本身仍下降。"
        "这些差额同时包含跳过机会与账户净值、整手路径变化，不能机械等同被拒绝段的原始利润之和。"
        "主历史年化收益低于买入持有，较早仍高于买入持有。统一长期趋势过滤对两个时期的效果不同，不能据此认为任意按年份切换就已有效。"
        "不反向选择均线条件救回本设定。下一项用同一持仓状态下固定版与最新可用版模型的负面预测分歧建立退出否决，先确认时钟和版本。")
    next_path = ROOT / "docs/510300_MODEL_UPDATE_VETO_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300长期趋势进入资格_第147轮_20260909/长期趋势资格_结果及全部中文规则.md",
        "长期趋势决定整段持仓机会的进入资格", "CLOSED_EARLY_SHARPE_GAIN_MAIN_COLLAPSE_ONE_EARLY_POINT_TARGET_INSUFFICIENT",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "MODEL_UPDATE_VETO_INPUT_REVIEW", "比较同一128当日持仓状态的原固定预测与最近已保存模型预测，研究负面更新分歧退出")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=148, registered=False, planned_settings=1, planned_new_accounts=4)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=147,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO147")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
