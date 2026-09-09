"""交付当前强弱退出的负面增量，接续不重训的持仓模型版本检验。"""
import json
from research.session_strength_within_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第127轮新增当前日内隔夜强弱没有稳定改善。主基础／压力净夏普0.769／0.719，年化6.23%／5.78%，最大回撤17.02%／17.24%；"
        "较早净夏普0.764／0.740，年化8.69%／8.39%，最大回撤13.79%／13.92%。原114主净夏普0.884／0.831、较早0.752／0.728。"
        "较早略有改善，但主历史明显退步，四项夏普均未达1.2。关闭本次加入当前六十日强弱值的方法，不改其窗口、变化项、交互、模型惩罚或退出阈值挽救。")
    detail = ("六项必要测试4.29秒；114次九因子拟合及四个完整账户的运行计时4.65秒，不含开发、测试、文档及保存结果核对。"
        "141个原模型时点中114次成熟拟合完成、27个原支持不足，无求解失败或未成交。主每档20周期、320持仓判断，其中320行有模型；"
        "较早每档9周期、265持仓判断，其中61行有模型，204行沿用可知价格保护。不能把较早这些无模型时期说成新九因子已有长期预测支持。"
        "已核114个保存方程、1461条参考状态、1170个实际持仓状态、762个有效预测、58个完整周期及24个分年增量记录。"
        "两档成本合计8条周期退出时间变化。主基础2021年2月3日进入的一笔，原2月19日退出盈利11215.21元，本轮3月1日追踪保护退出亏5459.66元；"
        "2026年6月一笔晚一天退出有利。较早2017年一笔提前退出不利，2019年一笔延后有利，不据此为某年份选规则。"
        "主基础终值比114少14099.70元，价格和分红贡献合计少14517.30元，费用反而少417.60元；较早基础终值多2416.68元。"
        "这是保存成交路径的差异，不是另跑零费用账户，也不把后续金额变化都归为同一笔费用。"
        "下一项拟固定单笔持仓首次收盘选择的原114模型版本，无需重训。已核原主每档3周期16状态、较早每档2周期47状态存在实质参数变化，"
        "合计126状态，最大系数或截距变化约0.02033；仅是旧路径参数比较，没有新固定版本预测或账户。"
        "完整目标与独立证据仍未达到。")
    next_path = ROOT / "docs/510300_ENTRY_VINTAGE_EXIT_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300日内隔夜强弱周期内退出_第127轮_20260909/日内隔夜强弱周期内退出_结果及全部中文规则.md",
        "日内隔夜强弱补充周期内退出", "COMPLETED_SESSION_STRENGTH_MAIN_DETERIORATION_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "EXISTING_WITHIN_CYCLE_MODEL_VINTAGE_PREFLIGHT_COMPLETE_IMPLEMENTATION_PENDING",
        "原114单笔实际持仓固定首次持仓收盘模型版本，零新拟合直接比较完整账户")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=128, registered=False,
        preflight_receipt="reports/research/510300_entry_vintage_exit_preflight_20260909/result.json", new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
