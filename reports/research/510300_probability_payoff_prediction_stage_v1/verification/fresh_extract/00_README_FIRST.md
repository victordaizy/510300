# 阅读导航

先读训练结果.md和stage_decision.json。当前优先顺序见authority_update.json：先概率与盈亏幅度，之后逐步增加节点；每年至少5次仍是最终策略要求。

protocol.json及freeze.json固定本轮训练和比较规则。inputs含124个月末样本、10节点、上一轮保存参数及预测、标签复算所需原行情和来源回执。models为本轮新增的444个条件均值拟合结果，results提供660条完整预测、分项误差和固定对照的前后半段比较。

运行code/probability_payoff_prediction_stage_v1.py的verify命令，--root指定解压目录，可检查成熟标签、保存模型最优化方程、固定概率及所有预测误差；该入口不拟合、不增加节点、不运行账户。依赖版本见requirements.txt。

本轮仅评价预测，不生成账户、频率、夏普或回撤新结果。原已关闭账户仍原样保留，不能将本轮误差下降自动转换成账户达标。
