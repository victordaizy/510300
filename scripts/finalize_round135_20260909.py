"""关闭失败的月度历史排名，交付全部中文规则并转入固定协方差比较。"""
import json
from research.recent_reference_selection_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第135轮未改善并结束这一设定。按过去242个交易日净夏普月度选择132、91或现金，"
        "主基础／压力净夏普0.661／0.613，年化3.03%／2.79%，最大回撤8.08%／8.78%；"
        "较早净夏普0.673／0.633，年化4.01%／3.76%，最大回撤6.84%／7.03%。"
        "四项夏普和年化收益均低于132及固定各半134，四项中三项整段年化收益低于买入持有。"
        "较早回撤小于132及134，但未转化为更高夏普。过去表现排序未在此组合中解决阶段适应；不据此断言所有动态策略无效。"
        "不调整本轮窗口、月份、现金门槛或参考挽救；完整目标未实现。")
    detail = ("六项必要合成测试4.03秒，四条新账户核心2.4021077秒，零新模型、零新参考和下载；核心秒数不含开发、测试、核对、文档。"
        "从原基础账户独立复算两段月首窗口、评分及选择，确认两费用共享评分，所选目标仍匹配各自费用，末日开盘未进入收盘窗口。"
        "共核对5646个判断、四条实际账户及65个完整周期。主各80次月首尝试、8次选择改变，实际各23周期、253个持股收盘、56次成交；"
        "较早各60次尝试、9次改变，基础9周期244持股收盘31成交，压力10周期245持股收盘33成交。主有266个正目标原点，不等于253个实际持股收盘。"
        "相对134，主基础价格分红收益少21038.30元、费用少219.94元，净少20818.36元；主压力净少20379.86元。"
        "较早基础价格分红少10551.80元、费用少794.22元、净少9757.58元；较早压力净少12381.96元。"
        "失败主要反映新选择路径损失了收益机会，不能归因于手续费比固定各半更多。"
        "原父模型的114个成熟月度版本属于已有研究档案；下文完整列出使用因子、模型含义及进出规则，本轮没有重新生成模型参数包。")
    next_path = ROOT / "docs/510300_COVARIANCE_REFERENCE_PAIR_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300历史表现月度选择_第135轮_20260909/月度选择_结果及全部中文规则.md",
        "按历史净夏普月度选择两策略或现金", "COMPLETED_RECENT_SELECTION_WEAKER_THAN132_AND134_CLOSED_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "COVARIANCE_REFERENCE_PAIR_PLANNED", "复用既有最小方差月度预算，检验132与91的共同波动能否改善实际组合")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=136, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
