"""导出既有训练结果和期权资料；不训练、不回测、不下载。"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300_训练数据与期权资料_20260924"
BUILD = ROOT / "outputs/01a0c884-9648-77a2-b339-c9fd5ea97c5f/data_options_20260924"
TRAIN = OUT / "训练数据"
OPT = OUT / "期权数据"
A = ROOT / "reports/research/510300_dense_probability_payoff_nodes_v1"
B = ROOT / "reports/research/510300_node_distribution_calibration_v1"
SOURCE_OPTION = ROOT / "data/raw/return_tail/options"
SHEETS: list[dict] = []
SOURCES: dict[str, list[dict]] = {"训练数据": [], "期权数据": []}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_source(src: Path, dst: Path, package: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise ValueError(f"复制内容不一致：{src.name}")
    SOURCES[package].append({
        "导出文件": dst.relative_to(OUT / package).as_posix(),
        "项目来源": src.relative_to(ROOT).as_posix(),
        "源文件SHA256": sha256(src), "处理": "原样复制",
    })


def csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sheet(name: str, title: str, note: str, frame: pd.DataFrame,
          widths: list[int] | None = None, dates: list[str] | None = None,
          percentages: list[str] | None = None) -> None:
    clean = json.loads(frame.to_json(orient="split", date_format="iso", force_ascii=False))
    SHEETS.append({"name": name, "title": title, "note": note,
                   "headers": clean["columns"], "rows": clean["data"],
                   "widths": widths or [18] * len(frame.columns),
                   "dates": dates or [], "percentages": percentages or []})


def export_parquet(src: Path, folder: Path, package: str) -> pd.DataFrame:
    frame = pd.read_parquet(src)
    copy_source(src, folder / src.name, package)
    target = folder / f"{src.stem}.csv"
    csv(frame, target)
    SOURCES[package].append({"导出文件": target.relative_to(OUT / package).as_posix(),
                             "项目来源": src.relative_to(ROOT).as_posix(),
                             "源文件SHA256": sha256(src), "处理": "Parquet 转为 UTF-8 BOM CSV；列名和数值保持来源定义"})
    return frame


def main() -> None:
    for folder in [TRAIN, OPT, BUILD]:
        folder.mkdir(parents=True, exist_ok=True)

    for src_root, prefix in [(A, "01_日样本训练与扩点"), (B, "02_节点分布校正")]:
        for src in sorted((src_root / "results").glob("*.csv")):
            copy_source(src, TRAIN / prefix / "结果" / src.name, "训练数据")
        for src in sorted((src_root / "inputs").iterdir()):
            if src.is_file():
                if src.suffix == ".parquet":
                    export_parquet(src, TRAIN / prefix / "输入", "训练数据")
                else:
                    copy_source(src, TRAIN / prefix / "输入" / src.name, "训练数据")
        for filename in ["protocol.json", "summary.json", "prediction_summary.json", "阶段结果.md", "协议字段说明.json"]:
            src = src_root / filename
            if src.exists():
                copy_source(src, TRAIN / prefix / "说明" / src.name, "训练数据")

    market = pd.read_parquet(A / "inputs/market.parquet")
    samples = pd.read_csv(A / "results/全部成熟日样本.csv")
    predictions = pd.read_csv(A / "results/完整逐日预测.csv")
    triggers = pd.read_csv(A / "results/逐日节点触发表.csv")
    selected = triggers[triggers.S1_PLUS_FORECAST_ONSET.astype(str).str.lower().eq("true")].copy()
    if (len(market), len(samples), len(predictions), len(selected)) != (3476, 2510, 14091, 148):
        raise ValueError("原数据行数发生变化，请检查来源后导出。")
    csv(selected, TRAIN / "便于查看/148个原始候选节点.csv")
    calibrated = pd.read_csv(B / "results/完整节点预测.csv")
    corrected = calibrated[calibrated.model.eq("TYPE_OFFSET_CALIBRATION")].copy()
    merged = selected.merge(corrected[["month", "p_win", "conditional_gain", "conditional_loss", "expected_net_proxy", "joint_gate_pass"]],
                            on="month", validate="one_to_one", suffixes=("_原始", "_校正"))
    cols = ["month", "entry_date", "exit_date", "if_event", "new_only", "p_win_原始", "conditional_gain_原始",
            "conditional_loss_原始", "expected_net_proxy_原始", "p_win_校正", "conditional_gain_校正",
            "conditional_loss_校正", "expected_net_proxy_校正", "target_net_proxy", "target_profit", "joint_gate_pass_校正"]
    nodes = merged[cols].copy()
    nodes.columns = ["信号日期", "计划入场日期", "标签结束日期", "原IF节点", "新增节点", "原预测胜率", "原盈利幅度",
                     "原亏损幅度", "原预测净期望", "校正预测胜率", "校正盈利幅度", "校正亏损幅度", "校正预测净期望",
                     "事后ETF净收益代理", "事后代理是否盈利", "校正是否通过门槛"]
    csv(nodes, TRAIN / "便于查看/148个节点_预测与事后结果对照.csv")

    all_metrics = []
    for prefix, root in [("第一轮", A), ("第二轮", B)]:
        frame = pd.read_csv(root / "results/账户联合评价.csv")
        frame.insert(0, "研究轮次", prefix)
        frame.insert(1, "是否复用前轮账户", (prefix == "第二轮") & frame.model.eq("UNCHANGED_PREDICTIONS"))
        all_metrics.append(frame)
    comparisons = pd.concat(all_metrics, ignore_index=True)
    csv(comparisons, TRAIN / "便于查看/全部36组账户比较_含4组复用.csv")

    cases = [("原节点", A, "DENSE_IF_VOL_GAMMA", "S0_IF_ONLY"),
             ("增加节点", A, "DENSE_IF_VOL_GAMMA", "S1_PLUS_FORECAST_ONSET"),
             ("节点校正", B, "TYPE_OFFSET_CALIBRATION", "S1_PLUS_FORECAST_ONSET")]
    overview, cycles, years = [], [], []
    for label, root, model, stage in cases:
        for filename, collector in [("账户联合评价.csv", overview), ("完整持仓周期.csv", cycles), ("完整自然年交易次数.csv", years)]:
            d = pd.read_csv(root / "results" / filename)
            d = d[d.model.eq(model) & d.node_stage.eq(stage) & d.scenario.eq("STRESS")].copy()
            d.insert(0, "方案", label)
            collector.append(d)
    overview = pd.concat(overview, ignore_index=True)
    cycles = pd.concat(cycles, ignore_index=True)
    years = pd.concat(years, ignore_index=True)
    csv(overview, TRAIN / "便于查看/三组主要结果_压力成本.csv")
    csv(cycles, TRAIN / "便于查看/三组主要结果_63条成交记录.csv")
    csv(years, TRAIN / "便于查看/三组主要结果_逐年次数.csv")

    inventory = []
    descriptions = {
        "510300_contract_master.parquet": "合约主表；含认购、认沽和已调整合约；主表快照不等于逐日调整台账",
        "510300_tushare_eod.parquet": "日开高低收、结算、成交量及持仓；不含连续历史买卖报价",
        "510300_sse_risk_indicators.parquet": "交易所风险指标；含 Delta、Gamma、Theta、Vega、隐含波动率",
        "510300_sse_daily_statistics.parquet": "510300 期权每日成交和持仓汇总",
        "sse_trading_calendar.parquet": "上交所交易日历",
    }
    for filename, description in descriptions.items():
        d = export_parquet(SOURCE_OPTION / filename, OPT / "完整数据", "期权数据")
        date_col = "trade_date" if "trade_date" in d else "list_date"
        dates = d[date_col].dropna().astype(str)
        inventory.append({"数据表": filename.removesuffix(".parquet"), "行数": len(d), "日期字段": date_col,
                          "起始日期": dates.min(), "结束日期": dates.max(), "用途与限制": description})
    inventory = pd.DataFrame(inventory)
    csv(inventory, OPT / "数据清单.csv")
    copy_source(A / "inputs/dividends.csv", OPT / "完整数据/ETF现金分红.csv", "期权数据")
    for src in [ROOT / "reports/data_quality/510300_option_chain_feasibility_v1.json",
                ROOT / "config/510300_option_chain_feasibility_v1.yaml"]:
        copy_source(src, OPT / "已有数据说明" / src.name, "期权数据")

    definitions = [
        ("month", "日样本/预测", "字段沿用旧名，实际为信号日期，并非只含月末。"),
        ("decision_at / label_end_at", "日样本/预测", "决策时点与未来标签结束时点；训练只纳入结束时间早于当次模型时点的标签。"),
        ("entry_date / exit_date", "日样本/预测", "次日开盘入场与第10个持有交易日收盘退出对应日期；这里只是标签，不保证账户成交。"),
        ("target10", "事后标签", "持有区间末日收盘与持有期现金分红相对首日开盘的毛收益。"),
        ("target_net_proxy", "事后标签", "target10 减 0.0024 的固定成本代理；不是逐笔账户收益。"),
        ("target_profit", "事后标签", "target_net_proxy 大于0取1，否则取0。"),
        ("p_win", "预测", "模型预测净收益代理为正的概率；不是已证明的胜率。"),
        ("conditional_gain / conditional_loss", "预测", "盈利条件下的预计收益幅度、亏损条件下的预计亏损幅度；两者均用正数表示。"),
        ("expected_net_proxy", "预测", "p_win × conditional_gain − (1 − p_win) × conditional_loss。"),
        ("joint_gate_pass", "节点筛选", "p_win>0.5、conditional_gain>conditional_loss、expected_net_proxy>0 的共同状态。"),
        ("S0_IF_ONLY / S1_PLUS_FORECAST_ONSET", "节点筛选", "原IF节点集合 / 原IF节点加预测共同状态由假转真的新增节点集合。"),
        ("model / node_stage / scenario", "结果分组", "模型、节点阶段、成本情景共同定义一个账户；不同账户的交易可能重复。"),
        ("BASE / STRESS", "成本", "单边佣金2基点且最低5元；单边滑点分别5 / 10基点。"),
        ("net_sharpe", "账户指标", "20万元全账户、含空仓日，按242交易日年化；无风险与现金利息设0。"),
        ("win_rate", "账户指标", "完整持仓周期中净盈利周期所占比例。"),
        ("realized_cash_payoff_ratio", "账户指标", "盈利周期平均净赚金额 / 亏损周期平均净亏金额；不同于单笔收益率盈亏比。"),
        ("max_drawdown", "账户指标", "正数表示最大回撤幅度；逐日账簿中的 drawdown 则为负数。"),
        ("completed_cycles_by_entry_year", "年度频次", "按入场年份统计完整空仓→持仓→空仓周期；完整自然年要求至少5次。"),
        ("complete_calendar_year", "年度频次", "2019—2025为完整自然年；2018和2026为部分年份，不混同验收。"),
        ("contract_code / option_type", "期权", "合约代码按文本读取；C为认购，P为认沽。"),
        ("strike / contract_unit", "期权", "行权价 / 每张对应ETF份数；分红调整可能改变两者，不可全历史固定10000份。"),
        ("close / settlement", "期权", "收盘价 / 结算价；均不等同于指定时点实际可成交买卖价。"),
        ("turnover_10k_cny", "期权", "成交金额单位为万元；其余价格、行权价通常为元/ETF份。"),
        ("retrieved_at", "所有原始来源", "本地获取时间，不是该指标在历史决策时点已公开可用的证明。"),
    ]
    dictionary = pd.DataFrame(definitions, columns=["字段", "适用数据", "含义"])
    csv(dictionary, TRAIN / "字段说明.csv")
    csv(dictionary[dictionary.适用数据.isin(["期权", "所有原始来源"])], OPT / "字段说明.csv")

    train_readme = """# 510300 训练数据交付（2026-09-24）

这是已完成的两轮研究的数据导出，不包含新训练、新回测或新增行情采集。

- 第一轮：2510个成熟日样本；2013个预测日×7模型=14091条预测；148个候选节点（10个原节点+138个新增节点）；24组账户。
- 第二轮：同样148个节点×3模型=444条预测；8组新账户和4组复用账户。两轮合并36组比较记录只代表32组新账户。
- ETF行情3476行，2012-05-28至2026-09-11；信号截止2026-08-12；账户2018-05-02至2026-08-26，包含空仓日。
- 原始输入、全部结果按研究轮次分目录保留。Parquet原文件和CSV转换同时提供，CSV使用UTF-8 BOM便于Excel打开。
- “便于查看”有148节点预测/事后标签对照、三组主要结果、63条成交记录、逐年频次、全部账户比较。63条交易属于三个不同方案，不能视为63个独立交易机会。
- 候选节点不等于已成交交易；预测概率不等于实现胜率；未来标签不可用于当天选股或入场。
- 金额盈亏比与收益率盈亏比分开。当前三组主要结果均为压力成本情景，不代表最优策略。
- 全账户本金20万元，净夏普目标1.3，最大回撤目标10%，每个完整自然年至少5次完整交易。当前没有同时达标的账户。
- 历史已多次研究；滚动时间顺序预测并不使这段历史重新成为独立、未见过的留出检验。
- 第一轮协议中继承的未使用 account_end=2026-08-18，实际按已声明的最后信号第十个持有日结束，日期为2026-08-26。原协议未覆盖；请同时阅读“协议字段说明.json”。
- “完整逐日账户”和“完整持仓周期”包含多个模型、节点阶段和成本情景；按 model、node_stage、scenario 分组后分析，不可混为同一账户。
- 期权资料单独打包。本轮未把期权加入现有账户，未估算任何期权策略的胜率、盈亏比、回撤或夏普。

## 建议查看顺序
1. 便于查看/三组主要结果_压力成本.csv
2. 便于查看/148个节点_预测与事后结果对照.csv
3. 便于查看/三组主要结果_63条成交记录.csv 与逐年次数表
4. 各轮结果中的完整预测和逐日账户
5. 原始输入与字段说明

需要完整模型参数和可复核代码时，可使用此前的“510300_概率盈亏训练与逐步扩点_两轮结果_20260924.zip”。
"""
    (TRAIN / "先读我.md").write_text(train_readme, encoding="utf-8")
    option_readme = """# 510300 期权既有数据（2026-09-24导出）

本包是已有文件的原样复制和CSV转换，不是新采集，不包含策略训练或新的样本绩效检验。

包括2874条合约主表、194040条日行情、191270条风险指标、1611条每日汇总和1611条交易日历，以及ETF现金分红表。日行情与风险指标覆盖2019-12-23至2026-08-14；最新获取时间在2026-08-16，不能当作2026-09-24的最新市场报价。

数据均围绕上交所510300 ETF期权。IO指数期权不是同一合约，本包不含IO数据。认购和认沽都保留，option_type为C/P。

## 用于研究时的具体限制
- 合约主表含610条已调整合约。合约单位和行权价需要按历史生效日处理；当前项目仍没有单独的完整历史调整台账及对应证据文件。主表快照不能替代逐日调整映射。
- EOD表没有连续历史bid/ask；close和settlement不能直接当作有保证的成交价。需要把买卖价差、滑点、手续费、成交量和可退出性纳入研究。
- 风险指标的历史同日公开时点尚未由本包证实；旧可行性配置采用保守滞后一日。retrieved_at只是本地获取时间。
- 附带的旧可行性报告使用较严格的完整期权曲面标准；其中16.82%的曲面通过比例不等于只有16.82%的日行情有效。
- 本包没有导出Sina终端“最后若干日”分钟文件及362行保留盘口快照；它们不构成连续全历史可成交报价。分钟文件跨很多合约，整体日期最小/最大值不代表每个合约持续覆盖。
- 旧研究的完美未来方向情景属于事后上界，不是可以交易的信号。本包不把它作为期权策略通过证据。
- 若研究买入认沽，标签应为事前固定合约选择规则下实际期权扣费损益，不能用ETF收益取负或1减ETF胜率代替。
- 每个合约代码请按文本导入；原始Parquet保留原schema，CSV便于直接查看。全部文件保留已有来源和获取时间列。

官方规则参考（本次只查阅规则，不获取新行情）：
- 上交所期权投资与准入：https://one.sse.com.cn/onething/qqtz/
- 510300期权基本条款：https://www.sse.com.cn/assortment/options/contract/c/c_20230303_5717360.shtml
- 2026年1月合约调整：https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml
"""
    (OPT / "先读我.md").write_text(option_readme, encoding="utf-8")

    guide = pd.DataFrame([
        ["用途", "查看两轮训练的既有输入、预测、节点与成交结果；另附已有510300期权资料。"],
        ["账户目标", "20万元；成本后夏普≥1.3；最大回撤≤10%；每个完整自然年至少5次完整交易。"],
        ["当前结果", "尚无同时达标账户。历史已多次研究，不属于全新独立留出样本。"],
        ["行情与样本", "3476行ETF日行情；2510个成熟日样本；原始行情截至2026-09-11。"],
        ["预测与节点", "第一轮14091条预测；148个候选节点；第二轮444条节点预测。"],
        ["比较记录", "36组账户比较含4组复用；合计32组新账户。不同方案成交不能合并计数。"],
        ["成本口径", "主要结果均为STRESS：单边佣金2基点且最低5元，加单边10基点滑点。"],
        ["账户区间", "2018-05-02至2026-08-26，2022个交易日；含所有空仓日。"],
        ["训练标签", "ETF十日毛收益减0.24%固定成本代理；不是账户实际净收益。"],
        ["交易频次", "按入场年统计完整持仓周期；2019—2025为完整自然年。"],
        ["盈亏比", "主要结果使用平均盈利金额÷平均亏损金额；原CSV另有收益率盈亏比。"],
        ["完整数据", "完整预测、特征和逐日账户见训练CSV数据包；本工作簿提供便于查看的选定表。"],
        ["期权资料", "194040条日行情和191270条风险指标；截至2026-08-14；不是今日市场报价。"],
        ["期权边界", "尚未加入当前账户；本次没有期权训练或期权策略绩效。历史报价与调整台账限制见期权包。"],
    ], columns=["项目", "说明"])
    sheet("数据说明", "510300｜训练数据与期权资料", "2026-09-24导出；静态记录，不随手动修改其他工作表重新计算。", guide, [22, 110])
    metric_map = {"方案": "方案", "opportunities": "完成交易数", "win_rate": "实现胜率", "realized_cash_payoff_ratio": "金额盈亏比",
                  "net_sharpe": "净夏普", "max_drawdown": "最大回撤", "net_profit_cny": "净利润（元）", "annual_frequency_pass": "每完整年至少5次"}
    sheet("三组主要结果", "三组主要结果｜压力成本", "同一20万元账户口径；三组均未达到联合目标。来源：便于查看/三组主要结果_压力成本.csv。",
          overview[list(metric_map)].rename(columns=metric_map), [18, 15, 16, 16, 14, 16, 20, 22], percentages=["实现胜率", "最大回撤"])
    compare_map = {"研究轮次": "研究轮次", "是否复用前轮账户": "复用账户", "model": "模型", "node_stage": "节点阶段", "scenario": "成本情景",
                   "opportunities": "交易数", "win_rate": "胜率", "realized_cash_payoff_ratio": "金额盈亏比", "net_sharpe": "净夏普",
                   "max_drawdown": "最大回撤", "net_profit_cny": "净利润（元）", "all_numeric_targets_met": "联合目标达标"}
    sheet("全部账户比较", "36组比较记录｜含4组复用", "全部方案保留。复用账户不是新增试验；不得把不同模型的成交叠加为同一账户。",
          comparisons[list(compare_map)].rename(columns=compare_map), [14, 14, 37, 33, 14, 12, 14, 16, 14, 16, 20, 18], percentages=["胜率", "最大回撤"])
    sheet("148个候选节点", "148个节点｜预测与事后标签", "候选节点不等于已成交。幅度均为收益率；事后字段不可用于当时决策。完整特征见CSV。",
          nodes, [17, 18, 18, 15, 15] + [19] * 11, dates=["信号日期", "计划入场日期", "标签结束日期"], percentages=list(nodes.columns[5:14]))
    cycle_map = {"方案": "方案", "entry_date": "入场日期", "exit_date": "退出日期", "shares": "ETF份数", "entry_fill_price": "买入成交价",
                 "exit_fill_price": "卖出成交价", "net_pnl_cny": "净损益（元）", "holding_sessions": "持有交易日", "exit_reason": "退出原因"}
    sheet("三组成交明细", "三组方案｜63条成交记录", "交易属于三个不同方案，可能重叠；所有金额已按对应压力成本扣费。",
          cycles[list(cycle_map)].rename(columns=cycle_map), [18, 18, 18, 15, 18, 18, 20, 18, 36], dates=["入场日期", "退出日期"])
    year_map = {"方案": "方案", "year": "入场年份", "complete_calendar_year": "完整自然年", "completed_cycles_by_entry_year": "完整交易次数", "at_least_five": "达到5次要求"}
    sheet("三组年度次数", "逐年频次｜部分年份单列", "2018和2026不完整，是否达到5次的字段为空；频次要求按每个完整自然年分别判断。",
          years[list(year_map)].rename(columns=year_map), [18, 16, 20, 20, 22])
    market_map = {"date": "日期", "open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价",
                  "dividend": "每份现金分红", "total_simple": "日总收益率", "wealth": "总收益累积值"}
    sheet("ETF日行情", "510300｜3476行已有日行情", "2012-05-28至2026-09-11；价格为元/份。总收益累积值保留来源数值，未另行重置基期。",
          market[list(market_map)].rename(columns=market_map), [18, 17, 17, 17, 17, 20, 19, 22], dates=["日期"], percentages=["日总收益率"])
    sheet("期权数据清单", "510300期权｜已有文件清单", "完整CSV和Parquet见期权数据包。日行情与风险指标截至2026-08-14；尚未回测加入后的账户。",
          inventory, [44, 17, 18, 18, 18, 76], dates=["起始日期", "结束日期"])

    (BUILD / "workbook_data.json").write_text(json.dumps({"sheets": SHEETS}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    receipt = {"导出日期": "2026-09-24", "新增训练": 0, "新增回测": 0, "新增行情采集": 0,
               "期权合约数": 2874, "期权日行情行数": 194040, "期权风险指标行数": 191270,
               "工作簿各表数据行数": {s["name"]: len(s["rows"]) for s in SHEETS}, "压缩包": []}
    for package in [TRAIN, OPT]:
        csv(pd.DataFrame(SOURCES[package.name]), package / "数据来源.csv")
        records = []
        for p in sorted(package.rglob("*")):
            if p.is_file() and p.name != "FILE_INDEX.csv":
                records.append({"文件": p.relative_to(package).as_posix(), "字节数": p.stat().st_size, "SHA256": sha256(p)})
        csv(pd.DataFrame(records), package / "FILE_INDEX.csv")
        target = OUT / f"510300_{package.name}_20260924.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for p in sorted(package.rglob("*")):
                if p.is_file():
                    archive.write(p, p.relative_to(package).as_posix())
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad is not None or len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError(f"压缩包校验失败：{target.name}")
        receipt["压缩包"].append({"文件": target.name, "字节数": target.stat().st_size, "SHA256": sha256(target), "CRC": "通过"})
    (OUT / "导出记录.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
