"""交付九次价格准备事件失败，并转向保存候选薄弱情景诊断。"""
import json
from research.setup_nine_reversal_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第130轮九次相对弱势反弹与强势退出未达目标。主基础／压力净夏普0.310／0.290，年化3.20%／2.95%，最大回撤34.31%／34.59%；"
        "较早净夏普0.059／0.048，年化−0.47%／−0.66%，最大回撤26.38%／26.53%。四账户年化收益均低于买入持有。"
        "关闭这套固定四日比较九次首次事件交易，不改变比较天数、计数、方向、事件定义或追加保护挽救，也不改成十三次倒计时继续试。")
    detail = ("七项合成测试3.58秒，零拟合、零新参考，四账户及完整指标核心运行1.9196515秒，不含开发、测试、保存核对及文档。"
        "3456日完整保留，3452个有效四日比较，42个弱势九次事件和45个强势九次事件。主每档12周期、796个持股收盘；较早每档7周期、494个持股收盘，"
        "评价中无缺失、未成交或锁定重试。已用首尾价格与分红调整的独立等价式及按同方向分组长度，核对全部四日比较和首次事件；"
        "共用模块检查5646个实际判断、四账户及38个完整周期，没有重新模拟。"
        "主基础保存价格和分红收益50748.10元、费用4254.51元、净盈利46493.59元；主压力净盈利42447.39元。"
        "较早基础价格分红先亏2437.70元，再付2212.12元，净亏4649.82元；压力净亏6596.84元。"
        "主基础2021年3月5日买入直到2022年6月13日才退出，持有307个收盘、净亏42637.08元；"
        "2023年8月至2024年2月一笔再亏16132.26元。较早2015年6月至8月、2018年6月至2019年1月各亏约2.46万和2.48万元。"
        "弱势达到九次没有证明下跌结束，而等待强势九次也可能长期不能退出，进入和退出均缺乏稳定优势。"
        "下一步先比较已保存候选的较弱情景与费用影响，集中后续研究；诊断不新训练或回测，不算第131轮新策略。")
    next_path = ROOT / "docs/510300_SAVED_CANDIDATE_FRONTIER_DIAGNOSTIC_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300九次价格比较反弹_第130轮_20260909/九次价格比较反弹_结果及全部中文规则.md",
        "九次相对弱势反弹与强势退出", "COMPLETED_SETUP_NINE_REVERSAL_REJECTED_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "SAVED_CANDIDATE_FOUR_SCENARIO_DIAGNOSTIC_PROTOCOL_READY",
        "只读已有候选的主历史与较早历史两档费用，按薄弱情景寻找改进重点")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=131, registered=False, diagnostic_only=True, new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
