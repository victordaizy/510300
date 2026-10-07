"""说明待评价月份补齐的实际范围及仍缺少的本地资料。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_local_pending_m1_completion_v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def text(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main():
    summary, boundary = load(OUT / "summary.json"), load(OUT / "data_boundary.json")
    predictions = pd.read_csv(OUT / "results/保存模型的历史重建输出.csv")
    labels = {"A_PRICE": "价格基准 A", "B_PRICE_SURPRISE": "加入预期差 B", "MATURE_MEAN_REFERENCE": "成熟训练均值"}
    table = ["| 原保存模型 | 五日毛收益模型输出 | 成熟训练事件数 |", "|---|---:|---:|"]
    for row in predictions.itertuples():
        table.append(f"| {labels[row.model]} | {row.prediction_gross_5d_return:.4%} | {row.n_train} |")
    text("研究结论.md", f"""
# 已有行情补齐待评价月份：完成历史模型输出，完整标签仍缺四日

本轮找到了比最近训练输入多三个交易日的本地行情，并据此补齐新版M1研究中2026年8月事件的观察日特征与3条保存模型输出。价格基准、加入预期差和均值参考的五日毛收益输出均为负；实际五日结果仍缺9月17、18、21、22日行情，不能计算这一事件的完整预测误差或新增成熟评价样本。模型与参数没有重拟合，原失败结论保留，当前目标仍是20万元、净夏普1.3、最大回撤10%。

## 补齐了什么

最近几轮共同输入有3476个交易日，至2026-09-11。本地原固定续行目录另存3479行行情，至2026-09-16，增加9月14、15、16日；旧3476日的开高低收价格完全一致，分红覆盖回执至9月16日。本轮只读取这些已有文件，未恢复旧续行策略。

原事件实际公布上界为9月14日17:00。按原规则，公布后首个完整交易日即9月15日收盘形成特征，历史假设入场日为9月16日，五日评价末日为9月22日。原待处理记录因使用9月11日行情而写为观察日价格缺失；本轮已经解决观察日输入，余下是评价期行情不足。

{chr(10).join(table)}

这是原模型在该历史特征上的算术输出，单位是五日毛收益率，不是已实现收益，也不是今天的买卖建议。三份模型均只使用16个成熟事件，最后训练标签于2026-08-24收盘成熟；本轮仅代入保存的中心、尺度、系数与截距。

## 时间顺序和证据性质

父模型文件实际创建于2026-09-22 20:37，晚于本次重建的9月15日历史决策时点。即使所用训练标签和特征在历史时点之前已经可知，也不能把今天才计算的输出伪装成9月15日已经发布的预测。因此，本轮新增独立前向预测数为0，输出类型明确标记RETROSPECTIVE_RECONSTRUCTION_NOT_PRECOMMITTED_FORECAST。

本轮价格输入虽含9月16日，但特征只使用至9月15日收盘。将9月15日之后行情全部截去后，三条输出完全不变。原16个成熟事件的六项价格/消息特征也已重算，最大差异为{boundary['maximum_old_feature_error']:.3g}。

## 当前尚缺什么

固定五日窗口是9月16、17、18、21、22日，本地只有首日。其余四日缺失保留为PENDING_COMPLETE_HORIZON，不用首日收益代替五日结果，不填0，不压缩持有期。

其他已登记路径也作了区分：吸收率末尾月的标签是原协议截至8月14日所造成的截尾，其8月28日行情其实已在本地，不能写成缺行情；IF信号本身止于8月12日，单独增加ETF价格不能补出IF新信号；LPR和私募已有事件的标签已经成熟，多三个价格日没有增加这些来源的事件月份。

上述核对只覆盖列明的训练分支和本地来源，不宣称所有潜在研究方向都无效。现有本地行情结束早于9月22日新模型的实际创建日期，因此无法凭这些数据建立这些模型创建后的独立表现。原已终止策略曾有的3个新增观察日记录仍原样保留，不能转移给本轮新模型当独立证据。

## 本轮状态

完成1个历史事件特征、3条保存模型输出及其时序核对；新增拟合0、回测账户0、下载0、成熟评价标签0、独立前向预测0。未运行策略账户，也没有据此生成当前仓位。

用户已明确不用采集，采集任务保持关闭。当前可计算部分已经完成，余下完整标签与独立表现需要本地资料发生实际变化；既有历史上的重复训练不能替代这部分证据。没有运行中的本轮训练任务，也不把旧run.lock或过期等待状态当成仍在计算。
""")
    text("00_README_FIRST.md", """
# 交付导航

先读研究结论.md；实际输出在results/保存模型的历史重建输出.csv，特征在results/补齐的历史事件特征.csv。data_boundary.json给出三日增量、所需五日窗口及四个缺失日期。

inputs包含原保存模型、原待处理事件、原16个成熟事件、两份本地行情、分红和日历，以及原关闭结论与来源回执。freeze.json固定计算输入。protocol.json说明本轮是事后历史重建，不是事前已发布的预测。

用本地Python环境运行code/local_pending_m1_completion_v1.py的verify子命令，并以--root指定解压目录，即可只从包内文件复算。该入口不拟合、不下载、不运行账户。requirements.txt列出环境依赖。文件大小和SHA-256见FILE_INDEX.csv。

local_runtime_observation.json记录检查时任务禁用状态与匹配研究入口的Python进程，不是持久后台等待句柄。goal_continuation_state.json说明当前进度及未解决的数据条件；它不会修改系统目标状态。
""")
    text("用户需求.md", """
# 当前需求

只交易510300和现金，20万元、最大回撤10%，净夏普已按用户最新消息改为1.3。允许一年只有四五次机会，不凑次数。用户要求直接使用已有数据训练，不用采集。

此前已实际训练的模型保持原参数。本轮为已有本地资料补齐待处理月份的计算，不因采集关闭而下载新行情，也不把历史重建输出冒充独立验证。
""")
    text("GPT审阅提示词.md", """
请审阅这一有限补齐是否准确：

- 9月11日与9月16日两份本地行情是否旧前缀一致？新增三日是否确实足够计算9月15日观察特征，却不足9月16—22日完整五日标签？
- 三条输出是否只代入保存系数，没有重新拟合、改变窗口或使用观察日之后价格？原16个特征的复算是否一致？
- 模型实际创建于9月22日，是否明确将输出标为9月15日历史重建，而没有冒充事先发布的预测或独立前向证据？
- 是否保留未知实际收益和缺失四日，未用首日、0或短窗口替代？
- 是否将吸收率的协议截尾与行情缺失区分，将IF信号缺失与ETF价格延长区分？

请给出具体文件及问题。新目标为1.3，不是原1.5。本轮不申请恢复原失败模型、旧终止策略或采集任务。
""")
    text("EXCLUSIONS.md", """
# 复核范围

本包自含本轮特征和保存模型输出计算所需的直接数值输入。父级研报PDF和原网页不重复打包；它们及抽取过程在inputs/parent_delivery_receipt.json所指原包中。本轮没有重新验证研报抽取或重建父级训练，只核对保存数值的计算和时序。

不包含模型新拟合、新账户、新下载、真实订单或已实现五日结果。观察窗口之外价格只用于确认数据范围，不进入模型输出。根目录交付回执在ZIP外产生，验证解压目录和缓存不入包。没有外部GPT审阅。
""")
    state = {
        "recorded_at": summary["completed_at"],
        "previous_goal_turn_classification": "PROGRESS_FULL_EARLIER_ARCHIVE_REASSESSED_AT_1_3",
        "current_goal_turn_classification": "PROGRESS_LOCAL_INPUT_FOUND_AND_PENDING_FEATURES_SCORED",
        "completed_work": "补齐一个历史事件特征和三条保存模型输出；明确剩余四日标签缺口。",
        "target_net_sharpe": 1.3,
        "unresolved_condition": "NO_NEW_COMPLETE_LOCAL_EVALUATION_AND_POST_CREATION_EVIDENCE_WITH_COLLECTION_DISABLED",
        "same_condition_consecutive_goal_turns": 1,
        "condition_scope": "当前已登记训练分支与该待评价事件；不宣称所有潜在模型都无效。",
        "missing_label_dates": boundary["missing_horizon_dates"],
        "latest_local_market_source": load(OUT / "freeze.json")["source_paths"]["latest_prices.parquet"],
        "latest_local_market_date": boundary["latest_market_end"],
        "parent_model_created_at": boundary["parent_models_created_at"],
        "new_collection_enabled": False,
        "current_training_handles_confirmed_live": [],
        "next_step_when_data_changes": "先重新确认真实本地文件、数据时钟与完整标签；一个新增事件不自动证明夏普或恢复失败模型。",
        "goal_status_left_active": True, "goal_achieved": False, "orders_authorized": False,
    }
    (OUT / "goal_continuation_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("补齐结果及剩余数据条件已记录；原模型和旧结论保持。")


if __name__ == "__main__":
    main()
