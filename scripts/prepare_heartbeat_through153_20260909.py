"""同步第153轮已完成证据和第154轮明确方案，保留持续任务字段。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 153 and not index["goal_achieved"], "第153轮已完成状态不同")
    previous = (ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH152_20260909.md").read_text(encoding="utf-8")
    prefix = previous.split("下一153先读", 1)[0]
    old_progress = prefix.split("上一回合为PROGRESS：", 1)[1].split("\n\n", 1)[0]
    new_progress = ("第153轮一设置四新账户、六项必要测试、因子和完整账户核对、关闭及中文交付已经完成。152及更早完成阶段保留，"
        "没有运行中的测试、账户或核对进程。154下一方案已写好但尚未实现、测试、冻结、生成目标或建账，有明确下一动作，无阻塞。"
        "不要重跑已完成prepare、freeze、run、verify、finalize、preflight，独占回执均存在。设置PYTHONIOENCODING=utf-8，"
        ".venv\\Scripts\\python.exe -m scripts.read_510300_fast_status可读简短状态。权威reports/research/510300_sharpe_1_2_latest_research.json"
        "现为153轮、437不同设置、453已评价来源版本、458登记含5旧未运行、1970主绩效记录，goal_achieved=false。"
        "THROUGH152及更早接续点过时，不能据此重跑153。")
    prefix = prefix.replace(old_progress, new_progress, 1).replace("最后比较轮次152。", "最后比较轮次153。")
    new = """第153轮结果：research/vortex_risk_v1.py，输入research/vortex_risk_inputs_v1.py，config/510300_vortex_risk_v1.json，reports/research/510300_vortex_risk_v1。PRIMARY=VORTEX_RISK。六测试4.95秒通过，四账户核心1.7613448000047356秒，零训练、零新参考、零外部下载。核心不含实现、测试、保存核对及交付。规则固定相邻经济高低价14日正负移动和／共同TR和，正方向目标min(1,.10/vol20)，已知非正0，方向未知或正向vol20未知／不正NaN，原.1缓冲，下一真实OPEN。完整中文docs/510300_VORTEX_RISK_V1.md，原14日窗口／方向／父来源均不再改变或救回。

153主BASE／STRESS：S.1664843386557658／.017258720775419588，CAGR.009908191440092943／−.0016232829045388082，MDD−.25809400105722863／−.28802330864375897；早S.40543822899476983／.24836619835803356，CAGR.030832748245461672／.017428299392027315，MDD−.16699764805182413／−.17652473198062912。四CAGR均低BH，四S及MDD均劣143；状态CLOSED_VORTEX_RISK_FULL_GOAL_NOT_MET。主各79周期806持仓CLOSE，212／211成交；早各66周期699持仓，171／171成交。主807正797零目标，早700正519零，两费用目标完全相同，研究区间0未知、0方向相等。

153保存路径损益：主BASE价格加分红31386、佣金滑点17879.6387、净13506.3613元；STRESS29949.3／32091.35092／−2142.05092元。早BASE47682.5／14625.8489／33056.6511元；STRESS45815.6／27628.97268／18186.62732元。相对143终值主−52895.71134／−64891.88512，早−25757.82176／−43173.23368元。这是保存路径分解，不是零费用重放。频繁方向切换与交易成本侵蚀均存在，不将其改成确认143、不改窗口、不开阈值搜索。

153 scripts/verify_round153_20260909.py用当前财富除当前rawclose+div的等价比例独立还原经济OHLC，逐个14日math.fsum窗口和20日原始收益样本波动核对，经济价格不用于成交。四真实账户5646全部收盘、290实际周期、765真实OPEN成交已核对；成本、份额、权益和终点已还原，保存对照与原文件逐列一致。回执2026-09-09T16:42:34.710793+08:00，reviewer SHA53ed49e7c58b3dbec8965f2528dbcfb1b3b16641df016e002a1cbca0eab3358c，shared checker SHA2602e9268fc97f31f5af332ae45816947a10614b852e0d51c5900367db20dffc。交付deliverables/510300相邻高低价方向_第153轮_20260909/相邻高低价方向_结果及全部中文规则.md。scripts/finalize_round153_20260909.py已运行、索引更新，不重复。

下一154：先读docs/510300_BOUNDARY_REBALANCE_NEXT_20260909.md。已有143接受状态COMPLETED_FOUR_SHARPE_AND_RETURN_LOCAL_GAINS_EARLY_BELOW12_NO_INDEPENDENT_PROOF，仍作为未完全验证的均衡比较对象。保留其全部逐日目标、费用对应、初始买入、0全退及.1缓冲宽度；新检验只改变已有持仓越界后的调整终点，由目标中心改成最近边界。一个设置、四新实际账户、零训练／新参考／新外部行情。比较143、131、BH每段6保存+2新共8指标。未实现或冻结，直接推进必要实现与测试，有可执行下一步。

154每次15:05按自己cash+shares*rawclose+receivable算NAV，实际股票比例shares*rawclose/NAV。parent143未知→无新申请，已知0→全退，正目标且空仓→原中心目标首次买入。正目标已有持仓：L=max(0,parent-.1)，U=min(1,parent+.1)，位于[L,U]包括等号→保持；低于L→按L*NAV/rawclose/100向下整手得到目标份额；高于U→按U同理。只修改到最近边界，不再通过旧target_request二次套.1缓冲；selected execution weight与parent reference_weight应分开记录。整百份可能离边界不足一手，保留实际结果。未知不补0，0退出不受band；无附加锁或冷却，每日重算，终点优先OPEN全出。

154实现建议：保留所有旧冻结文件。新增boundary_request(account,close,parent,cfg)和独立账户入口。可以把research/event_clock_account_v1.py静态复制为新boundary_rebalance_account_v1.py，仅替换请求函数绑定并限定targets路径，真实execute_order、T+1、分红和每日账务保持原组件。新规则不应用于BUY_HOLD或prediction类型。原saved_target_batch_runner没有自定义账户参数，不能在冻结原文件上改；可新建同逻辑custom_account版本接受simulate_account函数，或单独154入口复制四账户框架。必须完整代码、明确中文日志，不用隐藏运行时代码注入。沿用parent143保存targets，不用其已成交份额代替事前目标。测试应包含中心vs边界两个方向、边界等号、空仓首次买、已知0全出、NaN保留、整手、真实nextOPEN进出及分红权益；必要时在新冻结前明确纯数值等号容差，不能看154收益再改。

154原143保存parent在reports/research/510300_trend_noise_reference_blend_v1/{evaluation,earlier_diagnostic}/{BASE,STRESS}/TREND_NOISE_REFERENCE_BLEND_decisions.parquet，字段origin_index/origin/execution_date/reference_weight；共同来源对齐可用saved_parent_target_alignment_v1.aligned_target_frames(data,parents_by_cost,[model],cfg,start)。其SHA已被config/510300_heikin_price_state_v1.json冻结。143/131/BH保存ledgers已在config/510300_vortex_risk_v1.json和更早绑定，三比较目录见research/vortex_risk_v1.CONTROLS。新规则要按自身NAV验证每个请求，旧saved_target_account_checks默认调至中心，不可直接把它当154请求已通过；可复用周期/净值/费用/绩效核对部分并独立实现边界数量验证。

方法文献已浏览https://arxiv.org/pdf/1204.6488，比例成本模型的区间外调至边界提供控制形式依据；本项目143不是已证明的无成本最优目标，固定.1宽度、初始中心买入、0全退、最低佣金/整手/跳跃也不同，绝不声称154是理论最优交易策略。不得改143因子、120趋势、20波动、父来源、带宽或费用；失败关闭此固定机制，不搜索边界比例或有利年份。不做新大包抽样为弱候选找正区间，仍需全目标和独立证据。

"""
    suffix = previous.split("共用行情", 1)[1]
    suffix = "共用行情" + suffix
    suffix = suffix.replace("无需参考目标的153可空parents，但先看批运行接口，保持完整calendar和成本身份。", "154需要143父目标，保持同成本身份和完整calendar；默认中心目标核对不适用于边界请求。")
    prompt = prefix + new + suffix
    target = ROOT / "docs/510300_RESEARCH_HEARTBEAT_THROUGH153_20260909.md"
    require(not target.exists(), "第153轮持续接续文档已存在")
    target.write_text(prompt, encoding="utf-8")
    old = tomllib.loads(Path("E:/CodexData/.codex/automations/510300-1-2/automation.toml").read_text(encoding="utf-8"))
    require(old["id"] == "510300-1-2" and old["kind"] == "heartbeat", "持续任务身份不同")
    args = {"id": old["id"], "mode": "update", "kind": old["kind"], "name": old["name"], "prompt": prompt.rstrip(),
            "rrule": old["rrule"], "status": old["status"], "targetThreadId": old["target_thread_id"]}
    if "notification_policy" in old:
        args["notificationPolicy"] = old["notification_policy"]
    path = ROOT / "reports/research/510300_heartbeat_update_through153_args.json"
    write_json(path, args, exclusive=True)
    print(json.dumps({"参数文件": str(path), "提示字数": len(args["prompt"]), "状态": old["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
