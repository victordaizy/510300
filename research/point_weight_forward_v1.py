"""新日期上的固定账户配对研究：只登记和测量，不生成实盘或券商指令。"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import point_weight_forward_inputs_v1 as logic
from research import upward_episode_anatomy_v1 as common
from research import daily_supply_test_v1 as risk_base
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_account_nr7_complement_v1 import verify_account
from research.point_forward_calendar_check_v1 import coverage_from_calendar
from research.point_forward_observer_v1 import check_program

OUT = ROOT / "reports/research/510300_point_weight_forward_v1"
OBSERVER = ROOT / "reports/research/510300_point_forward_observer_v1"
CALENDAR = ROOT / "data/reference/sse_trade_calendar_2026.csv"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    common.save_json(Path(path), value)


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT))


def local_imports(entries):
    """保留实际账户和信号模块的本地导入来源，不把外部库副本混入研究结果。"""
    waiting, found = list(entries), set()
    while waiting:
        name = waiting.pop()
        if name in found or not (ROOT / name).is_file():
            continue
        found.add(name)
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.module == "research":
                    modules = ["research." + x.name for x in node.names]
                elif node.module and node.module.startswith("research."):
                    modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [x.name for x in node.names if x.name.startswith("research.")]
            waiting.extend(x.replace(".", "/") + ".py" for x in modules)
    return sorted(found)


def source_state():
    check_program()
    state = read(OBSERVER / "state.json")
    return state, ROOT / state["version"]


def calendar_dates():
    frame = pd.read_csv(CALENDAR)
    if set(frame.calendar_year) != {2026}:
        raise ValueError("本接续版本只接纳已核对的2026年官方日历，不能臆造后续年份。")
    return pd.DatetimeIndex(pd.to_datetime(frame.trade_date))


def initialize():
    if OUT.exists():
        raise RuntimeError("配对研究已经登记，不重置起点或覆盖协议。")
    source, folder = source_state()
    dates = calendar_dates()
    origin = dates[dates > pd.Timestamp(source["last_known_close"])][0]
    execution = dates[dates > origin][0]
    frozen = common.now()
    if logic.clock(frozen) >= origin.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5):
        raise ValueError("第一新决策时刻已经发生，不能按旧边界登记。")
    paths = local_imports(["research/point_weight_forward_v1.py", "research/point_weight_forward_inputs_v1.py",
        "research/point_core_observation_inputs_v1.py", "research/point_monthly_model_inputs_v1.py",
        "research/point_account_nr7_inputs_v1.py", "research/point_weight_information_inputs_v1.py",
        "research/anti_overfit_evidence_inputs_v1.py"])
    OUT.mkdir(parents=True)
    code = {}
    for name in paths:
        destination = OUT / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = common.digest(destination)
    inputs = {}
    for name in ("inputs/candidate_prices.parquet", "inputs/dividends.csv", "inputs/ordinary_models.json", "inputs/within_models.json",
                 "results/完整候选意向.parquet", "results/完整实际登记.parquet"):
        destination = OUT / "seed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / name, destination)
        inputs[name] = common.digest(destination)
    shutil.copytree(folder / "inputs/config", OUT / "seed/inputs/config")
    for path in sorted((OUT / "seed/inputs/config").glob("*.json")):
        inputs[str(path.relative_to(OUT / "seed"))] = common.digest(path)
    protocol = {
        "study": "510300_POINT_WEIGHT_FORWARD_V1", "frozen_at": frozen,
        "role": "PREDECLARED_PROSPECTIVE_MECHANISM_COMPARISON",
        "question": "对当前同一固定点位A，保留已保存仓位大小是否在一段事前指定的新期间同时改善实际账户净收益和净夏普？",
        "primary": "SAVED_WEIGHT", "comparison": "POINT_BINARY", "candidate_source": logic.binary.PARENT_A,
        "design_last_close": source["last_known_close"], "first_origin": str(origin.date()),
        "first_execution": str(execution.date()), "first_origin_clock": (origin.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)).isoformat(),
        "selection_disclosure": "这两个版本已在全部历史中比较，近期改善、早期pB失败及不确定性均已知。首个新收盘前固定本次唯一比较；旧历史不并入新收益。",
        "complexity": "继承当前点位的9组状态和既定月度退出模型，未声称变成简单策略。首版新机制的低复杂度要求继续；本研究没有提出新信号或新增学习退出。",
        "legacy": "旧85/15混合策略及旧失败裁决保持终止。本研究不执行该旧混合账户，不恢复其历史有效性。",
        "capital_each_account": 200000, "assets": ["510300.SH", "CASH_CNY"], "costs": ["BASE", "STRESS"],
        "binary_rule": "完全沿用此前POINT_A账户：父目标正且空仓才进入，持有中只执行原风险减仓，父目标零或账户保护退出。",
        "weighted_rule": "完全沿用此前SAVED_WEIGHT：目标=min(0.5,原目标)，原10个百分点调整带，目标零全退、受阻全退锁定。",
        "account_rules": "20万元，最高50%股票预算、5日ES预算2.5%、10%跳空预算5%且受回撤余量限制；10%收盘回撤触发下一可卖开盘全退及停止新开仓。",
        "execution": "上一个收盘的信号和计划，次日开盘100份整手、0.001元不利刻度、T+1，开盘只按现金和风险缩减计划；分红登记、应收和到账分开。",
        "cost_details": {"BASE": {"commission": .0002, "slippage": .0005, "minimum": 5},
                         "STRESS": {"commission": .0004, "slippage": .001, "minimum": 5}},
        "annual_days": 252, "cash_and_risk_free_rate": 0, "terminal_sessions": 1008,
        "minimum_candidate_cycles": 30, "minimum_wins_and_losses": 5,
        "information_rationale": "1008个实际交易日约四年；30个完成周期和各至少5个盈亏只是最低信息量，不是保证足够的功效计算。若次数继续偏少，到期明确证据不足，不延期等到好看；这不是每年交易配额。",
        "evaluation_schedule": "只有第1008个实际交易日收盘后评价一次；中间季度末仅描述。未来日历以官方公布为准。",
        "economic_requirements": "两成本下，主版本净年化和净夏普均严格超过基准且为正，实际净pB>1、平均周期净收益>0、账户最大回撤<=10%。交易次数软目标，全部年份和等待时间同时列出。",
        "paired_blocks": [20, 252], "paired_replications": 2000, "trade_replications": 5000, "seed": 20261001,
        "uncertainty": "两种固定配对区块的夏普和年化增量2.5%分位均>0，年份组抽样的pB和均值下界分别>1、>0；任何未定义重复都不通过。有限年份的百分位区间仅是稳健性筛查，不保证精确覆盖率或消除全部过拟合。",
        "stop_rules": "到期仅作一次裁决；经济要求失败则拒绝，样本不足或区间不足则证据不足。违规时钟/缺日/改规则则独立性无效。不得改窗口、延迟终点、丢交易或重置账户救回。",
        "unclosed_positions": "到期按真实末收盘市值和应收计账户，不强平；未完成周期不计胜率。",
        "scope_limit": "配对研究检验一个资金分配增量，不保证增加交易次数，也不自动满足整个策略研究目标。",
        "registration": "每个新收盘冻结全部账户输入及算法版本，真实时间须在该收盘15:05之后、次开盘09:30之前；漏日不补为及时。",
        "input_maintenance": "当前接纳器只覆盖2026年日历和既有分红账本；遇新年历或新分红停止并保留记录。未来仅允许另留收据的非信号数据接口维护，原算法及已记账前缀不得改写，迟到日期不恢复及时资格。",
        "program_hashes": code, "seed_hashes": inputs, "orders_authorized": False,
        "background_service_started": False, "goal_achieved": False,
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "freeze.json", {"at": frozen, "protocol_sha256": common.digest(OUT / "protocol.json"),
                               "libraries": {"numpy": np.__version__, "pandas": pd.__version__}})
    save(OUT / "state.json", {"at": frozen, "status": "WAITING_FOR_FIRST_NEW_CLOSE", "registered_origins": 0,
        "observed_account_sessions": 0, "completed_account_cycles": 0, "latest_run": None, "goal_achieved": False})
    print("账户配对研究已事前登记；没有回测新历史账户，也没有新前瞻收益。")


def verify_frozen():
    protocol = read(OUT / "protocol.json")
    freeze = read(OUT / "freeze.json")
    if common.digest(OUT / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("事前比较协议改变。")
    if freeze["libraries"] != {"numpy": np.__version__, "pandas": pd.__version__}:
        raise ValueError("账户计算库版本变化，需保留原环境和已有结果。")
    for name, expected in protocol["program_hashes"].items():
        if common.digest(ROOT / name) != expected:
            raise ValueError("原计算程序改变，停止前瞻配对：" + name)
    for name, expected in protocol["seed_hashes"].items():
        if common.digest(OUT / "seed" / name) != expected:
            raise ValueError("冻结时的来源种子改变：" + name)
    return protocol


def write_table(folder, name, frame):
    destination = folder / (name + ".parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def matching_inputs(snapshot, prices, signals, dividends):
    old_prices = pd.read_parquet(snapshot / "prices.parquet")
    old_signals = pd.read_parquet(snapshot / "signals.parquet")
    try:
        pd.testing.assert_frame_equal(old_prices, prices.iloc[:len(old_prices)].reset_index(drop=True), check_exact=True)
        last = old_prices.date.iloc[-1]
        current = signals.loc[signals.origin.le(last)].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_signals, current, check_exact=True)
        # 当前上游不接纳新分红；新增事件维护须另立来源收据，不能静默忽略。
        if common.digest(snapshot / "dividends.csv") != common.digest(dividends):
            return False
    except AssertionError:
        return False
    return True


def inspect_or_record(record=False, advance=False):
    protocol = verify_frozen()
    if (OUT / "terminal_result.json").exists():
        print("本配对窗口已有固定终点结果，不接续或重开评价。")
        return read(OUT / "terminal_result.json")
    if advance:
        from research.point_forward_observer_v1 import check_or_update
        check_or_update(True)
    source, folder = source_state()
    dates = calendar_dates()
    last = pd.Timestamp(source["last_known_close"])
    prices = pd.read_parquet(folder / "inputs/candidate_prices.parquet")
    if prices.date.iloc[-1] != last:
        raise ValueError("实际价格末日与上游状态不符。")
    if logic.clock(common.now()) < last.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5):
        raise ValueError("来源包含尚未到完整收盘时刻的数据。")
    seeded = pd.read_parquet(OUT / "seed/inputs/candidate_prices.parquet")
    pd.testing.assert_frame_equal(seeded, prices.iloc[:len(seeded)].reset_index(drop=True), check_exact=True)
    for kind in ("ordinary", "within"):
        old_models = read(OUT / f"seed/inputs/{kind}_models.json")["models"]
        now_models = read(folder / f"inputs/{kind}_models.json")["models"]
        if now_models[:len(old_models)] != old_models:
            raise ValueError("既有月度模型记录改变。")
    for seeded_config in (OUT / "seed/inputs/config").glob("*.json"):
        if common.digest(seeded_config) != common.digest(folder / "inputs/config" / seeded_config.name):
            raise ValueError("固定信号参数发生变化。")
    signals = pd.read_parquet(folder / "results/完整候选意向.parquet")
    signals = signals.loc[signals.candidate.eq(logic.binary.PARENT_A)].reset_index(drop=True)
    dividends_path = folder / "inputs/dividends.csv"
    session_dates = dates[(dates >= pd.Timestamp(protocol["first_execution"])) & (dates <= last)]
    if len(session_dates) > protocol["terminal_sessions"]:
        last = session_dates[protocol["terminal_sessions"] - 1]
        prices = prices.loc[prices.date.le(last)].reset_index(drop=True)
        signals = signals.loc[signals.origin.le(last)].reset_index(drop=True)
        session_dates = session_dates[:protocol["terminal_sessions"]]
    stamp = pd.Timestamp(common.now()).strftime("%Y%m%d_%H%M%S_%f")
    destination = OUT / "runs" / stamp
    destination.mkdir(parents=True)
    if record and last >= pd.Timestamp(protocol["first_origin"]) and len(session_dates) < protocol["terminal_sessions"]:
        execution = dates[dates > last][0]
        registration = OUT / "registrations" / last.strftime("%Y-%m-%d")
        if registration.exists():
            print("该收盘已有真实账户登记，保留原时刻。")
        else:
            pending = registration.with_name(".pending_" + registration.name + "_" + stamp)
            pending.mkdir(parents=True)
            write_table(pending, "prices", prices)
            write_table(pending, "signals", signals)
            shutil.copyfile(dividends_path, pending / "dividends.csv")
            generated = common.now()
            save(pending / "receipt.json", {"origin": str(last.date()), "execution_date": str(execution.date()),
                "generated_at": generated, "source_version": source["version"],
                "protocol_sha256": common.digest(OUT / "protocol.json"),
                "input_hashes": {p.name: common.digest(p) for p in pending.iterdir() if p.is_file()},
                "initial_status": logic.registration_status(last, execution, generated, protocol["frozen_at"], protocol["design_last_close"]),
                "registration_scope": "四账户共同的完整信号、市场输入及冻结算法；开盘价格仅按既定执行规则使用。",
                "orders_authorized": False})
            pending.rename(registration)
    records = []
    for receipt in sorted((OUT / "registrations").glob("*/receipt.json")):
        if receipt.parent.name.startswith(".pending_"):
            continue
        item = read(receipt)
        item["snapshot_verified"] = all(common.digest(receipt.parent / name) == value for name, value in item["input_hashes"].items())
        item["same_protocol"] = item["protocol_sha256"] == common.digest(OUT / "protocol.json")
        item["matches_current_causal_prefix"] = matching_inputs(receipt.parent, prices, signals, dividends_path)
        records.append(item)
    coverage = logic.account_registration_coverage(dates, protocol["first_execution"], last, records, protocol)
    write_table(destination, "账户实际登记覆盖", coverage)
    parent_registry = pd.read_parquet(folder / "results/完整实际登记.parquet")
    parent_registry = parent_registry.loc[parent_registry.candidate.eq(logic.binary.PARENT_A)].copy()
    parent_cov, parent_status = coverage_from_calendar(signals, parent_registry, dates, protocol["first_execution"],
        last, "2026-12-31", (logic.binary.PARENT_A,))
    write_table(destination, "原点位官方日历覆盖", parent_cov)
    accounts, table = {}, pd.DataFrame()
    prior = read(OUT / "state.json")
    if len(session_dates):
        div = normalize_dividends(pd.read_csv(dividends_path))
        data, _ = common.features(prices, div)
        risks, membership = risk_base.risk_estimates(data)
        if len(membership) and not membership.label_exit_idx.le(membership.decision_idx).all():
            raise ValueError("账户风险估计包含未成熟标签。")
        parents = signals.pivot(index="origin", columns="candidate", values="target").reset_index()
        accounts = logic.accounts(data, div, parents, risks, protocol["first_execution"])
        metrics = []
        for (cost, policy), account in accounts.items():
            verify_account(account)
            if prior.get("latest_run"):
                previous = ROOT / prior["latest_run"] / "accounts" / cost / policy
                if (previous / "daily.parquet").exists():
                    logic.prefix_check({name: pd.read_parquet(previous / (name + ".parquet")) for name in ("daily", "trades")}, account)
            output = destination / "accounts" / cost / policy
            for name in ("daily", "trades", "orders", "decisions", "rejections"):
                write_table(output, name, account[name])
            save(output / "terminal.json", account["terminal"])
            write_table(output, "逐年与部分年统计", logic.annual_statistics(account, dates))
            metrics.append({"cost": cost, "policy": policy, **logic.measured_metrics(account, dates)})
        table = pd.DataFrame(metrics)
        write_table(destination, "完整账户累计指标", table)
    phase = logic.phase_status(len(session_dates), coverage, protocol)
    result = {"at": common.now(), "status": phase, "latest_source_close": str(last.date()),
        "observed_account_sessions": len(session_dates), "registered_origins": len(records),
        "new_account_evaluations": len(accounts), "source_calendar_coverage": parent_status,
        "minimum_information_is_power_guarantee": False, "goal_achieved": False, "orders_authorized": False,
        "latest_run": relative(destination), "independent_full_strategy_validation": "NOT_ESTABLISHED"}
    if len(session_dates) == protocol["terminal_sessions"]:
        result["terminal"] = logic.evaluate_terminal(accounts, coverage, parent_status["full_calendar_coverage"], protocol, dates)
        save(OUT / "terminal_result.json", result)
    save(destination / "summary.json", result)
    save(OUT / "state.json", result)
    print(json.dumps(common.clean(result), ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="固定新期间账户比较；所有结果仅作研究。")
    parser.add_argument("action", choices=("initialize", "check", "record", "advance"))
    args = parser.parse_args()
    if args.action == "initialize":
        initialize()
    else:
        inspect_or_record(record=args.action in {"record", "advance"}, advance=args.action == "advance")


if __name__ == "__main__":
    main()
