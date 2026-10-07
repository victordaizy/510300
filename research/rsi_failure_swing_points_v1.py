"""一条固定RSI失败摆动多头假说：全部自然点位、实际盈亏比和旧候选覆盖。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import rsi_failure_swing_point_inputs_v1 as engine
from research.adaptive_allocation_v1 import normalize_dividends
from research.historic_cycles_point_translation_v1 import statistics
from research.point_forward_observer_inputs_v1 import frequency_tables

OUT = ROOT / "reports/research/510300_rsi_failure_swing_points_v1"
SOURCE = ROOT / "reports/research/510300_point_current_observation_20261001"
MODEL = "RSI14_FAILURE_SWING_2R20"
RSI_URL = "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/rsi"
ATR_URL = "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr"
require = engine.require


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze():
    require(not OUT.exists(), "RSI失败摆动研究已登记，不覆盖。")
    files = {}
    sources = {"inputs/prices.parquet": SOURCE / "inputs/candidate_prices.parquet",
               "inputs/dividends.csv": SOURCE / "inputs/dividends.csv",
               "inputs/coverage.json": SOURCE / "inputs/candidate_dividend_coverage.json",
               "inputs/old_points.parquet": SOURCE / "results/全部自然点位.parquet"}
    code = [Path(__file__), Path(engine.__file__), Path(common.__file__),
            ROOT / "research/historic_cycles_point_translation_v1.py", ROOT / "research/point_state_reconstruction_v1.py",
            ROOT / "tests/test_rsi_failure_swing_point_inputs_v1.py"]
    sources.update({"code/"+p.name: p for p in code})
    for name, source in sources.items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(target)}
    protocol = {"study": "510300_RSI_FAILURE_SWING_POINTS_V1", "at": common.now(),
                "user_request": "如果用别的技术分析，能帮助我们找到其他盈亏点位吗？",
                "scope": "仅510300多头日线点位；前完整周MACD、量与波动只作背景记录，不追加筛选。",
                "motivation": "原长空仓主要由底层信号缺失形成，本轮独立探索另一种动量恢复结构。",
                "bounded_novelty_check": "research/config/docs中未检索到专门failure swing/失败摆动实现；普通RSI超卖、布林带收复、突破回踩、假跌破、锚定量价及云图已有旧实现。此检索不宣称穷尽所有数学等价规则。",
                "source": RSI_URL, "source_scope": "来源只支持RSI及较高低点后突破前高的失败摆动概念；下述参数和交易实现是本研究设定，盈利性需检验。",
                "indicator": "前向现金分红平移收盘价；Wilder RSI14以首14个价格变化的平均正负变化初始化，之后递推。全平为50、无下跌但有上涨为100。",
                "entry": "先低于30，再回到30上方；记录反弹RSI最高值；出现至少一根更低RSI后，回落全程严格高于30，再严格突破反弹最高RSI，在该收盘发一次信号。回落触及30取消原设置，重新低于30才可建新设置。",
                "structural_stop": "从RSI回落开始日至确认日、当时已知的最低平移低价。信号次日开盘原价加当日已除息平移必须高于失效线；否则取消，不延迟等更好价格。",
                "exit": "按实际入场开盘与固定结构低点的距离定义R；目标为入场价上方2R。收盘达到目标、跌到失效线或已持有20个收盘，下一开盘申请退出；受阻则保留退出请求。没有盘中理想止损成交。",
                "reentry": "每个信号只在次日开盘尝试一次；已有点位及当日正在退出时不接受新信号；末端不强平。",
                "cost": "固定约十万元原价名义份额；单边万四佣金、最低5元、千一滑点、0.001不利刻度、100份整数手及实际登记资格股息。",
                "evaluation": "从2015-01-05开始单一连续观察至2026-09-30；报告完整历史及按入场年划分2015—2019、2020—2026两个组。年度次数按退出年，未完成点位排除收益指标。",
                "gate": "实际净收益计算p、B和pB。两个时期都需pB>1且平均净收益>0才列历史候选；历史已复用，即使达线也不是独立验证。",
                "frequency": "全年平均次数、零笔年份、最长等待；2020年以来新规则的入场时点是否恰逢原两候选均空仓只作重合诊断，不能直接相加次数或当成合并策略。",
                "configuration_count": 1, "parameter_grid": False, "trainings": 0, "source_atr_context": ATR_URL,
                "orders_authorized": False, "goal_achieved": False}
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("RSI失败摆动的一套入场、结构失效、2R计划目标和20收盘退出规则已固定。", flush=True)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "已运行的RSI固定假说不得覆盖。")
    for name, item in read(OUT / "freeze.json")["files"].items():
        require(common.digest(OUT / name) == item["sha256"], "来源发生变化："+name)
        if name.startswith("code/"):
            require(common.digest(ROOT / item["source"]) == item["sha256"], "执行代码与固定版本不同。")
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "configuration_count": 1})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    coverage = read(OUT / "inputs/coverage.json")
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= str(prices.date.iloc[-1].date()), "股息覆盖不足。")
    require(common.digest(OUT / "inputs/dividends.csv") == coverage["distribution_file_sha256"], "股息来源不符。")
    data, _ = common.features(prices, dividends)
    rsi = engine.wilder_rsi(data.ac, 14)
    signals = engine.detect(data.date, rsi, data.al)
    points, events = engine.observe(data, dividends, signals, "2015-01-05", 20, 2.)
    points["candidate"] = MODEL
    for column in ("entry_date", "entry_origin", "exit_date", "exit_origin"):
        points[column] = pd.to_datetime(points[column])
    context = data[["date", "above_ema20", "daily_hist", "weekly_last_date", "weekly_hist", "relative_volume", "rv_ratio", "up_volume_balance5"]]
    points = points.merge(context, left_on="entry_origin", right_on="date", how="left", validate="many_to_one")
    complete = points.loc[points.status.eq("COMPLETE")]
    metrics = []
    for era, low, high in (("ALL_HISTORY", "2015-01-01", "2026-12-31"),
                           ("ENTRY_2015_2019", "2015-01-01", "2019-12-31"),
                           ("ENTRY_2020_2026", "2020-01-01", "2026-12-31")):
        sample = complete.loc[complete.entry_date.between(low, high)]
        metrics.append({"era": era, **statistics(sample.point_net_return), "mean_realized_net_r": sample.realized_net_r.mean(),
                        "evidence_class": "EXPLORATORY_REUSED_HISTORY", "independent_validation": "NOT_ESTABLISHED"})
    metrics = pd.DataFrame(metrics)
    annual = pd.DataFrame([{"year": year, "complete_year": year < 2026,
                            "completed_points": int(complete.exit_date.dt.year.eq(year).sum())} for year in range(2015, 2027)])
    _, gaps, frequency = frequency_tables(points, data, "2015-01-05")
    old_points = pd.read_parquet(OUT / "inputs/old_points.parquet")
    overlap = []
    for point in points.loc[points.entry_date.ge("2020-01-01")].itertuples():
        record = {"entry_date": point.entry_date, "exit_date": point.exit_date, "status": point.status,
                  "point_net_return": point.point_net_return}
        for candidate in old_points.candidate.unique():
            local = old_points.loc[old_points.candidate.eq(candidate)]
            held = (local.entry_date.le(point.entry_date) & (local.exit_date.isna() | local.exit_date.gt(point.entry_date))).any()
            record[candidate+"_held_at_entry"] = bool(held)
        record["both_old_candidates_flat_at_entry"] = not any(v for key, v in record.items() if key.endswith("_held_at_entry"))
        overlap.append(record)
    overlap = pd.DataFrame(overlap, columns=None if overlap else ["entry_date", "exit_date", "status", "point_net_return", "both_old_candidates_flat_at_entry"])
    for name, frame in (("完整RSI日状态", signals), ("全部自然点位", points), ("执行与退出事件", events),
                        ("实际净点位指标", metrics), ("逐年完成次数", annual), ("空仓间隔", gaps), ("频率摘要", frequency),
                        ("新旧点位入场重合", overlap)):
        save_table(name, frame)
    passed = bool(metrics.loc[metrics.era.ne("ALL_HISTORY"), "point_estimate_pass"].all())
    summary = {"study": "510300_RSI_FAILURE_SWING_POINTS_V1", "at": common.now(),
               "status": "HISTORICAL_LEAD_NOT_INDEPENDENTLY_VALIDATED" if passed else "FIXED_RULE_REJECTED_CURRENT_POINT_GATE",
               "hypothesis": "RSI14_BOTTOM_FAILURE_SWING", "configuration_count": 1,
               "signals_since_2015": int((signals.entry_signal & signals.date.ge("2015-01-01")).sum()),
               "completed_points": len(complete), "unfinished_points": int(points.status.eq("RIGHT_CENSORED").sum()),
               "metrics": metrics.to_dict("records"), "average_full_year_count_2015_2025": annual.loc[annual.complete_year, "completed_points"].mean(),
               "average_full_year_count_2020_2025": annual.loc[annual.year.between(2020, 2025), "completed_points"].mean(),
               "entries_2020_onward": len(overlap), "entries_with_both_old_candidates_flat": int(overlap.both_old_candidates_flat_at_entry.sum()),
               "historical_point_gate_pass": passed, "independent_validation": "NOT_ESTABLISHED",
               "necessary_tests_passed": 5, "new_models": 0, "new_prospective_points": 0, "orders_authorized": False, "goal_achieved": False}
    common.save_json(OUT / "summary.json", summary)
    write_report(summary, metrics, annual, points, overlap)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(summary, metrics, annual, points, overlap):
    eras = {"ALL_HISTORY": "完整历史", "ENTRY_2015_2019": "2015—2019入场", "ENTRY_2020_2026": "2020—2026入场"}
    lines = ["# 其他技术分析的独立样例：RSI失败摆动多头点位", "",
             "**本轮完成一条固定新假说的历史检验。" + ("两个时期的历史点值达到门槛，仍未建立独立验证。" if summary["historical_point_gate_pass"] else "这条固定规则没有同时满足两个时期的实际净点位门槛，保留失败结果，不调参补救。") + "**", "",
             "用户希望探索其他技术分析，以发现原信号之外的点位。旧库已有普通RSI超卖、波动带收复、通道突破、突破回踩、假跌破收回、低点锚定量价和一目云图。专门的RSI失败摆动在本次有界检索中未发现已有实现。", "",
             f"[Fidelity的RSI说明]({RSI_URL})将较高低点后突破前高描述为底部失败摆动，并提示强趋势下超买超卖可持续。这只支持技术概念，不证明510300交易优势。", "",
             "本轮固定RSI14、30超卖起点：先超卖，再反弹，再回落且RSI守住30，最后突破前次RSI反弹峰值。所有步骤逐日确认，不把后来形成的高低点前移。收盘确认后次日开盘尝试一次；持有中忽略新的入场事件。", "",
             "失效线为RSI回落开始到确认日的已知最低价格；按照真实入场开盘至失效线的距离设2R计划目标。价格收盘碰到失效线或目标，或持有满20个收盘，均在下一开盘尝试退出。没有假定日内能按止损价成交。计划2R是毛空间，实际净B由最终完整交易计算。", "",
             "费用统一为单边万四、最低5元、千一滑点和不利价格刻度，登记资格股息计入；固定约十万元名义份额。只评价标的点位，未做期权收益或完整投资账户评价。", "",
             "| 时期 | 完成点位 | 净胜率p | 实际净B | p×B | 平均净收益/笔 | 平均实际净R |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {eras[r.era]} | {r.n} | {r.p:.2%} | {r.b:.3f} | {r.product:.3f} | {r.mean:+.2%} | {r.mean_realized_net_r:.3f} |")
    lines += ["", "标准亏损单位期望仍为p×B−q，q为亏损率。p×B严格大于1是用户额外指定门槛；正均值或计划2R均不能替代该门槛。分期按入场日，不在2019年末人为结束持仓；全年次数则按自然退出年统计。", "",
              "| 退出年份 | 完成点位 |", "|---|---:|"]
    for r in annual.itertuples():
        lines.append(f"| {r.year}" + ("（截至9月30日）" if not r.complete_year else "") + f" | {r.completed_points} |")
    lines += ["", f"2020—2025完整年份平均每年{summary['average_full_year_count_2020_2025']:.2f}笔。2020年以来独立运行产生{summary['entries_2020_onward']}个进入时点，其中{summary['entries_with_both_old_candidates_flat']}个进入时点原两条候选均为空仓。这说明时间上可能有补充，但没有模拟合并持仓，不能把次数或收益直接相加。", "",
              f"期末未完成点位{summary['unfinished_points']}笔，单列并排除收益指标。所有历史已被项目使用，本轮是新假说探索，不是独立样本外验证。没有依据某个时期的好结果更换时间范围、RSI阈值、结构低点、2R倍数或持有上限。", "",
              "本轮5项必要测试通过：Wilder递推、摆动确认及未来不改前段、触及30取消原设置、结构失效后按实际次日开盘成交及末端不强平、开盘已破位取消。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定RSI失败摆动多头点位研究。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
