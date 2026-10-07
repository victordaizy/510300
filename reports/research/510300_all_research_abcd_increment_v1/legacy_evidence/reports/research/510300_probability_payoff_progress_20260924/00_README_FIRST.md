# 阅读顺序

先读训练进展与结果.md和continuation_status.json。目标未完成，本包没有将高胜率或分项误差下降当作策略达标。

01_日样本训练与扩点：2510个已有日样本、808个有效拟合头、138个新增日期、24个完整账户。

02_固定节点分布校准：固定148节点、868个有效拟合头、8个新账户加4个原样复用账户。

所有直接输入、冻结协议、实现修正、参数、逐日预测、节点、成交和年度次数都在相应目录。协议字段说明.json解释从父级继承但未使用的展示字段，保留原冻结协议。

离线复算：用Python执行第一目录code/dense_probability_payoff_nodes_v1.py verify --root 第一目录；第二目录执行code/node_distribution_calibration_v1.py verify --root 第二目录。只核对保存参数和结果，0拟合、0新账户、0采集。完整参数和包文件身份由根FILE_INDEX.csv索引。
