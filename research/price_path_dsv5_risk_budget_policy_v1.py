"""冻结 B1 预测的独立政策翻译；不导入或调用父训练器。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

MODEL_ID = "510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1"
SLUG = "510300_price_path_dsv5_risk_budget_policy_v1"
CONFIG = f"config/{SLUG}.yaml"
MANIFEST = f"config/{SLUG}_manifest.json"
REPORT = f"reports/research/{SLUG}"
DATA = f"data/research/{SLUG}"
FORWARD = f"data/forward/{SLUG}"
TZ = timezone(timedelta(hours=8))
LEVELS = np.array([0.25, 0.5, 0.75, 1.0])


class ContractError(RuntimeError):
    """冻结输入、时钟或阶段合同被违反。"""


def now() -> str:
    return datetime.now(TZ).isoformat()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(root: Path, path: str) -> dict:
    p = root / path
    return {"path": path, "bytes": p.stat().st_size, "sha256": sha256(p)}


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode:
        raise ContractError(f"Git检查失败：{result.stderr.strip()}")
    return result.stdout.strip()


def committed(root: Path, paths: list[str]) -> None:
    git(root, "ls-files", "--error-unmatch", "--", *paths)
    git(root, "diff", "--exit-code", "HEAD", "--", *paths)


def load_config(root: Path) -> dict:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["model_id"] != MODEL_ID or cfg["parent_b2_used"]:
        raise ContractError("模型身份或B2隔离合同不符")
    if cfg["policy"]["levels"] != LEVELS.tolist():
        raise ContractError("冻结四档仓位不符")
    return cfg


def verify(root: Path, *, require_git: bool = True) -> tuple[dict, dict]:
    cfg = load_config(root)
    manifest = read_json(root / MANIFEST)
    for spec in [*manifest["implementation"], *manifest["inputs"].values()]:
        if identity(root, spec["path"]) != spec:
            raise ContractError(f"冻结文件哈希不一致：{spec['path']}")
    if require_git:
        committed(root, [MANIFEST, *[s["path"] for s in manifest["implementation"]]])
    status = read_json(root / cfg["inputs"]["parent_status"])
    if status["FINAL_STATE"] != "REJECTED_FROZEN_CONSTITUENT_FRAGILITY_INCREMENT":
        raise ContractError("父实验冻结拒绝状态不符")
    if status["G2"] != "PASS" or status["G3"] != "FAIL":
        raise ContractError("父实验B1/B2预测门谱系不符")
    return cfg, manifest


def weight_from_ratio(q: float) -> float | None:
    if not math.isfinite(q) or q <= 0:
        return None
    # 精确中点：7/8、5/8、3/8；相等时向较低仓位归属。
    if q < 64.0 / 49.0:
        return 1.0
    if q < 64.0 / 25.0:
        return 0.75
    if q < 64.0 / 9.0:
        return 0.5
    return 0.25


def policy_targets(ratios: list[float], previous: float | None = None) -> list[float | None]:
    result = []
    for q in ratios:
        target = weight_from_ratio(float(q))
        if target is not None:
            previous = target
        result.append(previous)
    return result


def import_predictions(root: Path, cfg: dict) -> pd.DataFrame:
    """Parquet列裁剪在读取时执行；禁止先读全表再删除B2或实际DSV5。"""
    projection = cfg["prediction_projection"]
    frame = pd.read_parquet(root / cfg["inputs"]["predictions"],
                            columns=list(projection)).rename(columns=projection)
    if set(frame.columns) != set(projection.values()):
        raise ContractError("预测表出现合同外字段")
    for c in ["origin_date", "next_tradable_open", "horizon_end_date", "B1_training_end_date"]:
        frame[c] = pd.to_datetime(frame[c], errors="raise").dt.normalize()
    if frame.duplicated(["offset", "origin_date"]).any():
        raise ContractError("预测原点重复")
    if not (frame.B1_training_end_date <= frame.origin_date).all():
        raise ContractError("训练标签在预测原点尚未成熟")
    if not (frame.next_tradable_open > frame.origin_date).all():
        raise ContractError("不是下一交易日开盘")
    for c in ["B0_predicted_DSV5", "B1_predicted_DSV5"]:
        if not (np.isfinite(frame[c]) & frame[c].gt(0)).all():
            raise ContractError("不可变父预测存在非正或缺失数值")
    # 模型账本只读取B1的版本身份和训练截止元数据，不读取任何系数。
    metadata = pd.read_parquet(root / cfg["inputs"]["coefficients"],
        columns=["offset", "refit_id", "model_vintage_origin_date", "maximum_training_horizon_end"],
        filters=[("model", "==", "B1")]).drop_duplicates()
    merged = frame.merge(metadata, left_on=["offset", "B1_model_vintage_id"],
                         right_on=["offset", "refit_id"], validate="many_to_one", how="left")
    if merged.refit_id.isna().any():
        raise ContractError("B1模型版本在父模型账本中不存在")
    if not (pd.to_datetime(merged.model_vintage_origin_date) <= merged.origin_date).all():
        raise ContractError("模型版本晚于原点")
    if not (pd.to_datetime(merged.maximum_training_horizon_end) == merged.B1_training_end_date).all():
        raise ContractError("模型训练截止与预测账本不一致")
    frame = frame.sort_values(["offset", "origin_date"]).reset_index(drop=True)
    expected_counts = {0: 178, 1: 178, 2: 174, 3: 173, 4: 176}
    if frame.groupby("offset").size().to_dict() != expected_counts:
        raise ContractError("父实验五个offset样本数变化")
    primary = frame[frame.offset.eq(0)]
    if primary.groupby("era_id").size().to_dict() != {"ERA_1": 60, "ERA_2": 59, "ERA_3": 59}:
        raise ContractError("父实验主网格时代变化")
    for offset in range(1, 5):
        indices = frame.index[frame.offset.eq(offset)].to_numpy()
        for era, chunk in enumerate(np.array_split(indices, 3), start=1):
            frame.loc[chunk, "era_id"] = f"ERA_{era}"
    frame["q"] = frame.B1_predicted_DSV5 / frame.B0_predicted_DSV5
    frame["continuous_weight"] = np.clip(frame.q ** -0.5, 0.25, 1.0)
    frame["target_weight"] = frame.q.map(weight_from_ratio)
    frame["view_state"] = "VIEW_ALLOWED"
    return frame


def inspect_variation(frame: pd.DataFrame, cfg: dict) -> dict:
    primary = frame.loc[frame.offset.eq(0)].sort_values("origin_date")
    w = primary.target_weight.to_numpy(dtype=float)
    years = ((primary.horizon_end_date.max() - primary.next_tradable_open.min()).days + 1) / 365.25
    gross_changes = float(np.abs(np.diff(np.r_[0.0, w, 0.0])).sum())
    annual_roundtrips = gross_changes / 2.0 / years
    reductions = primary.assign(reduced=primary.target_weight.lt(1.0)).groupby("era_id").reduced.sum().to_dict()
    g = cfg["gates"]["G1"]
    checks = {
        "non100_origins_at_least_12": int((w < 1).sum()) >= g["non100_origins_min"],
        "at_least_two_eras_three_reductions": sum(n >= 3 for n in reductions.values()) >= g["eras_with_at_least_3_reductions_min"],
        "average_target_between_45_and_95pct": g["average_target_min"] <= float(w.mean()) <= g["average_target_max"],
        "weight25_fraction_at_most_20pct": float((w == .25).mean()) <= g["weight25_fraction_max"],
        "annual_equivalent_roundtrips_at_most_8": annual_roundtrips <= g["annualized_equivalent_roundtrips_max"],
        "quarter_weight_changes": bool(np.allclose(np.diff(w) * 4, np.round(np.diff(w) * 4), atol=1e-12, rtol=0)),
        "t_plus_1_schedule_feasible": bool(primary.next_tradable_open.is_unique and primary.next_tradable_open.is_monotonic_increasing),
    }
    return {"passed": all(checks.values()), "checks": checks,
        "primary_origin_count": len(primary), "non100_origins": int((w < 1).sum()),
        "average_target_weight": float(w.mean()), "weight25_fraction": float((w == .25).mean()),
        "weight_counts": {str(k): int(v) for k, v in primary.target_weight.value_counts().sort_index().items()},
        "era_reduction_counts": reductions, "annualized_equivalent_roundtrips": annual_roundtrips,
        "target_total_absolute_change_including_endpoints": gross_changes,
        "elapsed_years": years, "T_PLUS_1_CHECK": "SCHEDULE_PROOF_ONLY_ACTUAL_LEDGER_CHECK_AFTER_G1",
        "q_min": float(primary.q.min()), "q_max": float(primary.q.max()),
        "origin_start": primary.origin_date.min(), "origin_end": primary.origin_date.max()}


def run_gates(root: Path) -> dict:
    cfg, manifest = verify(root)
    registration = read_json(root / f"{FORWARD}/registration.json")
    if registration["manifest_sha256"] != sha256(root / MANIFEST):
        raise ContractError("必须先完成本版本严格前向登记")
    path = root / f"{REPORT}/G0_G1.json"
    if path.exists():
        raise ContractError("G0/G1已完成；读取现有回执，禁止重跑或修改映射")
    frame = import_predictions(root, cfg)
    variation = inspect_variation(frame, cfg)
    (root / DATA).mkdir(parents=True, exist_ok=True)
    target_path = root / f"{DATA}/B0_B1_policy_targets.parquet"
    if target_path.exists():
        raise ContractError("政策预测投影已存在")
    frame.to_parquet(target_path, index=False)
    result = {"MODEL_ID": MODEL_ID, "created_at": now(), "freeze_commit": git(root, "rev-parse", "HEAD"),
        "manifest_sha256": sha256(root / MANIFEST), "G0": {"passed": True,
        "parent_hashes_match": True, "parent_trainer_called": False, "B2_values_read": False,
        "actual_DSV5_values_read": False, "columns_imported": list(cfg["prediction_projection"].values())},
        "G1": variation, "RETURN_EVALUATION": "ALLOWED_ONCE_AFTER_G1_COMMIT" if variation["passed"] else "NOT_ALLOWED",
        "PORTFOLIO_RETURN_READS": 0, "MARKET_PRICE_VALUES_READ": 0, "SHARPE": "NOT_COMPUTED",
        "STATE": "PASS_G0_G1" if variation["passed"] else cfg["gates"]["G1"]["failure_state"],
        "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False,
        "target_identity": identity(root, f"{DATA}/B0_B1_policy_targets.parquet")}
    write_new(path, result)
    if not variation["passed"]:
        write_new(root / f"{REPORT}/status.json", {**result,
            "FAMILY_STATE": "FROZEN_NO_VIEW_INSUFFICIENT_POLICY_VARIATION_NO_RESCUE",
            "ECONOMIC_EFFICACY_TESTED": False, "FORWARD_MODE": "STATUS_ONLY_NO_NEW_POLICY_TARGET",
            "CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY": "NONE"})
    return result


def require_historical_gate(root: Path) -> tuple[dict, dict]:
    cfg, manifest = verify(root)
    gate = read_json(root / f"{REPORT}/G0_G1.json")
    if not gate["G1"]["passed"] or gate["STATE"] != "PASS_G0_G1":
        raise ContractError("G1未通过，禁止读取历史组合收益或市场价格")
    committed(root, [f"{REPORT}/G0_G1.json", f"{FORWARD}/registration.json"])
    if identity(root, gate["target_identity"]["path"]) != gate["target_identity"]:
        raise ContractError("G1政策目标文件已改变")
    if gate["manifest_sha256"] != sha256(root / MANIFEST):
        raise ContractError("G1清单不符")
    return cfg, manifest


def load_market(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(root / cfg["inputs"]["prices"], columns=["date", "open", "close", "symbol"])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values("date").reset_index(drop=True)
    if prices.date.duplicated().any() or not prices.symbol.eq("510300.SH").all():
        raise ContractError("市场日期或证券身份不符")
    if not np.isfinite(prices[["open", "close"]]).all().all() or not prices[["open", "close"]].gt(0).all().all():
        raise ContractError("市场价格存在缺失、非正或无穷值")
    dividends = pd.read_csv(root / cfg["inputs"]["dividends"],
        usecols=["symbol", "record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    if not dividends.symbol.eq("510300.SH").all():
        raise ContractError("分红证券不符")
    for c in ["record_date", "ex_date", "payment_date"]:
        dividends[c] = pd.to_datetime(dividends[c], errors="raise").dt.normalize()
    if not ((dividends.record_date < dividends.ex_date) & (dividends.ex_date <= dividends.payment_date)).all():
        raise ContractError("分红登记、除息或付款时钟不符")
    if not (np.isfinite(dividends.cash_dividend_per_share) & dividends.cash_dividend_per_share.ge(0)).all():
        raise ContractError("分红金额不符")
    if dividends.duplicated(["record_date", "ex_date", "payment_date"]).any():
        raise ContractError("分红事件重复")
    return prices, dividends


def p2_risk(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    cash = dividends.groupby("ex_date").cash_dividend_per_share.sum()
    r = np.log((prices.close + prices.date.map(cash).fillna(0)) / prices.close.shift())
    dsv = 252.0 / 20.0 * r.clip(upper=0).pow(2).rolling(20, min_periods=20).sum()
    median = dsv.expanding(min_periods=1).median()
    return pd.Series((dsv / median.where(median > 0)).to_numpy(), index=prices.date)


def make_schedule(predictions: pd.DataFrame, prices: pd.DataFrame, offset: int) -> pd.DataFrame:
    subset = predictions.loc[predictions.offset.eq(offset)].sort_values("origin_date").copy()
    dates = pd.DatetimeIndex(prices.date)
    for row in subset.itertuples():
        i = dates.get_indexer([row.origin_date])[0]
        if i < 0 or i + 5 >= len(dates) or dates[i + 1] != row.next_tradable_open or dates[i + 5] != row.horizon_end_date:
            raise ContractError("父预测执行/到期日期与市场交易日不一致")
    # 缺失原点没有新预测；模拟器自然延续先前合法目标。
    return subset


def simulate(prices: pd.DataFrame, dividends: pd.DataFrame, schedule: pd.DataFrame,
             targets: np.ndarray, slippage_bp: float, *, capital: float = 200000.0) -> pd.DataFrame:
    """整手、T+1、最低佣金与现金分红的逐日总财富账本。"""
    if len(schedule) != len(targets) or schedule.next_tradable_open.duplicated().any():
        raise ContractError("目标长度不符或同日多次执行")
    start = pd.Timestamp(schedule.next_tradable_open.iloc[0])
    end = pd.Timestamp(schedule.horizon_end_date.iloc[-1])
    market = prices.loc[prices.date.between(start, end)]
    if market.empty or market.date.iloc[0] != start or market.date.iloc[-1] != end:
        raise ContractError("模拟日期覆盖不完整")
    decisions = {pd.Timestamp(row.next_tradable_open): (float(targets[k]), k, row.era_id)
                 for k, row in enumerate(schedule.itertuples())}
    events = list(dividends.itertuples())
    entitled: dict[int, float] = {}
    receivables: dict[int, float] = {}
    cash, shares, target, prev_equity = capital, 0, None, capital
    origin_index, era = -1, "UNSET"
    rows = []
    slip = slippage_bp / 10000.0
    for row in market.itertuples():
        date = pd.Timestamp(row.date)
        available = shares
        bought = sold = 0
        paid = recognized = commission = slippage = rounding = 0.0
        for k, event in enumerate(events):
            if date == event.ex_date and entitled.get(k, 0) > 0:
                amount = entitled[k] * float(event.cash_dividend_per_share)
                receivables[k] = amount
                recognized += amount
            if date >= event.payment_date and k in receivables:
                amount = receivables.pop(k)
                cash += amount
                paid += amount
        pre_equity = cash + shares * row.open + sum(receivables.values())
        exec_price, requested = None, shares
        if date in decisions:
            proposed, origin_index, era = decisions[date]
            if math.isfinite(proposed):
                if not 0 <= proposed <= 1:
                    raise ContractError("仓位超出无杠杆范围")
                target = proposed
            if target is not None:
                current_weight = shares * row.open / pre_equity
                side = 1 if target > current_weight else -1
                exec_price = row.open * (1 + side * slip)
                requested = int(math.floor(target * pre_equity / (100 * exec_price))) * 100
                delta = requested - shares
                # 若整手舍入翻转方向，按真实买卖方向重新计算执行价和目标股数。
                if delta != 0 and (1 if delta > 0 else -1) != side:
                    side = 1 if delta > 0 else -1
                    exec_price = row.open * (1 + side * slip)
                    requested = int(math.floor(target * pre_equity / (100 * exec_price))) * 100
                    delta = requested - shares
                    if delta * side < 0:
                        delta = 0
                if delta > 0:
                    quantity = delta
                    while quantity > 0:
                        fee = max(quantity * exec_price * .0002, 5.0)
                        if quantity * exec_price + fee <= cash + 1e-9:
                            break
                        quantity -= 100
                    if quantity:
                        commission = max(quantity * exec_price * .0002, 5.0)
                        cash -= quantity * exec_price + commission
                        shares += quantity
                        bought = quantity
                elif delta < 0:
                    sold = min(-delta, available)
                    if sold != -delta:
                        raise ContractError("T+1卖出超过旧可用股数")
                    if sold:
                        commission = max(sold * exec_price * .0002, 5.0)
                        cash += sold * exec_price - commission
                        shares -= sold
                slippage = (bought + sold) * abs(exec_price - row.open)
                rounding = max(target * pre_equity - shares * exec_price, 0.0)
        if cash < -1e-7 or shares < 0 or shares % 100 or sold > available:
            raise ContractError("现金、整手或T+1合同被违反")
        for k, event in enumerate(events):
            if date == event.record_date:
                entitled[k] = shares
        equity = cash + shares * row.close + sum(receivables.values())
        rows.append({"date": date, "old_available_shares": available, "same_day_bought_shares": bought,
            "sold_shares": sold, "shares": shares, "cash": cash, "dividends": paid,
            "dividend_receivable_recognized": recognized, "dividend_receivable": sum(receivables.values()),
            "commission": commission, "slippage": slippage, "rounding_cash": rounding,
            "target_weight": target, "actual_weight": shares * row.close / equity,
            "open": row.open, "close": row.close, "execution_price": exec_price,
            "requested_shares": requested, "pretrade_open_equity": pre_equity,
            "equity": equity, "daily_return": equity / prev_equity - 1,
            "origin_index": origin_index, "era_id": era, "T_PLUS_1_VIOLATION": False})
        prev_equity = equity
    return pd.DataFrame(rows)


def sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or np.std(values, ddof=1) <= 0:
        return math.nan
    return float(np.sqrt(252) * np.mean(values) / np.std(values, ddof=1))


def metrics(ledger: pd.DataFrame, capital: float = 200000.0) -> dict:
    r = ledger.daily_return.to_numpy()
    equity = ledger.equity.to_numpy()
    peak = np.maximum.accumulate(np.r_[capital, equity])[1:]
    # 固定从首个执行日开始分五交易日块，末尾不满五日不混入周尾部比较。
    weeks = np.array([np.prod(1 + r[i:i + 5]) - 1 for i in range(0, len(r) - 4, 5)])
    worst_count = max(1, math.ceil(.05 * len(weeks)))
    years = ((ledger.date.iloc[-1] - ledger.date.iloc[0]).days + 1) / 365.25
    return {"net_sharpe": sharpe(r), "net_cagr": (equity[-1] / capital) ** (1 / years) - 1,
        "cumulative_net_return": equity[-1] / capital - 1,
        "downside_semivariance": 252 * float(np.mean(np.minimum(r, 0) ** 2)),
        "max_drawdown": float(np.max(1 - equity / peak)),
        "worst_5pct_week": float(np.sort(weeks)[:worst_count].mean()) if len(weeks) else math.nan,
        "average_actual_weight": float(ledger.actual_weight.mean()),
        "commission_cny": float(ledger.commission.sum()), "slippage_cny": float(ledger.slippage.sum()),
        "cash_dividends_paid_cny": float(ledger.dividends.sum()), "ending_equity": float(equity[-1]),
        "trade_legs": int(((ledger.same_day_bought_shares + ledger.sold_shares) > 0).sum()),
        "t_plus_1_violations": int(ledger.T_PLUS_1_VIOLATION.sum()), "days": len(ledger)}


def reduction(candidate: float, baseline: float) -> float:
    return 1 - candidate / baseline if baseline > 0 else math.nan


def era_excess(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    if not candidate.date.equals(baseline.date):
        raise ContractError("组合时代对比日期不一致")
    output = {}
    for era in ["ERA_1", "ERA_2", "ERA_3"]:
        mask = candidate.era_id.eq(era)
        output[era] = float(np.prod(1 + candidate.loc[mask, "daily_return"]) -
                            np.prod(1 + baseline.loc[mask, "daily_return"]))
    return output


def stress_concentration(p0: pd.DataFrame, p2: pd.DataFrame, p3: pd.DataFrame) -> dict:
    p0wealth = p0.equity.to_numpy()
    peak = np.maximum.accumulate(np.r_[200000.0, p0wealth])[1:]
    dd = 1 - p0wealth / peak
    pnl = np.diff(np.r_[200000.0, p3.equity]) - np.diff(np.r_[200000.0, p2.equity])
    active, start, episodes = False, None, []
    for i, drawdown in enumerate(dd):
        if not active and drawdown >= .10:
            active, start = True, i
        if active and (drawdown <= 1e-12 or i == len(dd) - 1):
            episodes.append({"start": p0.date.iloc[start], "end": p0.date.iloc[i],
                "increment_cny": float(pnl[start:i + 1].sum()), "ongoing": drawdown > 1e-12})
            active, start = False, None
    total = float(pnl.sum())
    largest = max([max(e["increment_cny"], 0) for e in episodes], default=0.0)
    return {"total_increment_cny": total, "episodes": episodes,
            "largest_positive_episode_share": largest / total if total > 0 else math.nan}


def bootstrap_lower(ledger: pd.DataFrame, cfg: dict) -> dict:
    segments = [s.daily_return.to_numpy() for _, s in ledger.groupby("origin_index", sort=True)]
    n = len(segments)
    block = cfg["statistics"]["block_origins"]
    rng = np.random.default_rng(cfg["statistics"]["seed"])
    values = []
    for _ in range(cfg["statistics"]["bootstrap_repetitions"]):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        values.append(sharpe(np.concatenate([segments[i] for i in indices])))
    valid = np.isfinite(values)
    return {"repetitions": len(values), "finite_repetitions": int(valid.sum()),
            "one_sided_90pct_lower": float(np.quantile(np.array(values)[valid], .10)) if valid.all() else math.nan,
            "block_origins": block}


def deflated_diagnostic(ledger: pd.DataFrame, manifest: dict) -> dict:
    """缺失完整试验预算和跨试验SR方差时，只输出显式诊断，不能通过正式门。"""
    r = ledger.daily_return.to_numpy()
    trial_count = manifest["trial_inventory"]["manifest_count_including_current"]
    n = len(r)
    s = float(np.mean(r) / np.std(r, ddof=1))
    normal = NormalDist()
    gamma = .5772156649015329
    null_max = ((1 - gamma) * normal.inv_cdf(1 - 1 / trial_count) +
                gamma * normal.inv_cdf(1 - 1 / (trial_count * math.e))) / math.sqrt(n - 1)
    skew, kurt = float(pd.Series(r).skew()), float(pd.Series(r).kurt()) + 3
    denom = 1 - skew * s + (kurt - 1) * s * s / 4
    probability = normal.cdf((s - null_max) * math.sqrt(n - 1) / math.sqrt(denom)) if denom > 0 else math.nan
    return {"status": "BLOCKED_DSR_GLOBAL_TRIAL_BUDGET_INCOMPLETE", "passed": False,
        "manifest_inventory_count_including_current": trial_count,
        "global_trial_count_verified_complete": False, "cross_trial_sharpe_variance_available": False,
        "normal_null_std_proxy_probability_DIAGNOSTIC_ONLY": probability,
        "source": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf"}


def permute_blocks(weights: np.ndarray, eras: np.ndarray, rng: np.random.Generator,
                   block: int = 13) -> np.ndarray:
    result = weights.copy()
    for era in np.unique(eras):
        indices = np.flatnonzero(eras == era)
        chunks = [weights[indices[i:i + block]] for i in range(0, len(indices), block)]
        result[indices] = np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])
    return result


def run_duration_mean(weights: np.ndarray) -> float:
    return len(weights) / (1 + np.count_nonzero(np.diff(weights)))


def placebo(prices: pd.DataFrame, dividends: pd.DataFrame, schedule: pd.DataFrame,
            p3: pd.DataFrame, p1: pd.DataFrame, cfg: dict) -> dict:
    st = cfg["statistics"]
    rng = np.random.default_rng(st["seed"] + 1)
    w = schedule.target_weight.to_numpy(dtype=float)
    eras = schedule.era_id.to_numpy()
    true_turnover = float(np.abs(np.diff(w)).sum())
    true_duration = run_duration_mean(w)
    true_exposure = float(p3.actual_weight.mean())
    true_excess = float((p3.equity.iloc[-1] - p1.equity.iloc[-1]) / 200000)
    values, attempts = [], 0
    while len(values) < st["placebo_repetitions"] and attempts < st["placebo_attempt_limit"]:
        attempts += 1
        permuted = permute_blocks(w, eras, rng, st["block_origins"])
        turnover = float(np.abs(np.diff(permuted)).sum())
        if abs(turnover - true_turnover) > max(.5, .25 * true_turnover):
            continue
        if abs(run_duration_mean(permuted) / true_duration - 1) > .25:
            continue
        result = simulate(prices, dividends, schedule, permuted, 5)
        if abs(float(result.actual_weight.mean()) - true_exposure) > .02:
            continue
        values.append(float((result.equity.iloc[-1] - p1.equity.iloc[-1]) / 200000))
    complete = len(values) == st["placebo_repetitions"]
    pvalue = (1 + sum(v >= true_excess for v in values)) / (1 + len(values)) if complete else math.nan
    return {"status": "COMPLETE" if complete else "BLOCKED_PLACEBO_MATCHED_PERMUTATIONS_INSUFFICIENT",
        "attempts": attempts, "accepted_permutations": len(values), "actual_excess_vs_P1": true_excess,
        "upper_tail_probability": pvalue, "passed": bool(complete and pvalue <= .10),
        "placebo_excess_quantiles": np.quantile(values, [.10, .50, .90]).tolist() if values else [],
        "origin_distribution_preserved_exactly": True,
        "duration_and_turnover_preserved_approximately_only": True}


def evaluate_portfolios(prices: pd.DataFrame, dividends: pd.DataFrame,
                        schedule: pd.DataFrame, cfg: dict) -> dict:
    w3 = schedule.target_weight.to_numpy(dtype=float)
    q2 = p2_risk(prices, dividends).reindex(schedule.origin_date).to_numpy()
    w2 = np.array(policy_targets(q2), dtype=float)
    p3 = simulate(prices, dividends, schedule, w3, 5)
    p1_weight = float(p3.actual_weight.mean())
    hold_schedule = schedule.iloc[[0]].copy()
    hold_schedule.loc[:, "horizon_end_date"] = schedule.horizon_end_date.iloc[-1]
    ledgers = {}
    for cost, slip in [("base", 5), ("stress", 10)]:
        ledgers[cost] = {
            "P0": simulate(prices, dividends, hold_schedule, np.array([1.]), slip),
            "P1": simulate(prices, dividends, schedule, np.repeat(p1_weight, len(schedule)), slip),
            "P2": simulate(prices, dividends, schedule, w2, slip),
            "P3": p3 if cost == "base" else simulate(prices, dividends, schedule, w3, slip),
        }
    return {"ledgers": ledgers, "metrics": {cost: {p: metrics(l) for p, l in group.items()}
            for cost, group in ledgers.items()}, "P1_fixed_ex_post_target": p1_weight,
            "P2_target_weights": w2}


def economic_gates(evaluated: dict, cfg: dict) -> dict:
    b, s = evaluated["metrics"]["base"], evaluated["metrics"]["stress"]
    l = evaluated["ledgers"]["base"]
    eras = era_excess(l["P3"], l["P2"])
    concentration = stress_concentration(l["P0"], l["P2"], l["P3"])
    g2 = {
        "downside_reduction_vs_P1_at_least_20pct": reduction(b["P3"]["downside_semivariance"], b["P1"]["downside_semivariance"]) >= .20,
        "drawdown_reduction_vs_P1_at_least_20pct": reduction(b["P3"]["max_drawdown"], b["P1"]["max_drawdown"]) >= .20,
        "worst5pct_week_vs_P1_improves": b["P3"]["worst_5pct_week"] > b["P1"]["worst_5pct_week"],
    }
    g3 = {
        "base_sharpe_at_least_1p20": b["P3"]["net_sharpe"] >= 1.20,
        "sharpe_increment_vs_P2_at_least_0p15": b["P3"]["net_sharpe"] - b["P2"]["net_sharpe"] >= .15,
        "cumulative_excess_vs_P1_positive": b["P3"]["cumulative_net_return"] > b["P1"]["cumulative_net_return"],
        "cagr_at_least_4pct": b["P3"]["net_cagr"] >= .04,
        "cagr_shortfall_vs_P0_at_most_2pp": b["P3"]["net_cagr"] >= b["P0"]["net_cagr"] - .02,
        "drawdown_reduction_vs_P0_at_least_25pct": reduction(b["P3"]["max_drawdown"], b["P0"]["max_drawdown"]) >= .25,
        "two_positive_eras_vs_P2": sum(v > 0 for v in eras.values()) >= 2,
        "latest_era_vs_P2_positive": eras["ERA_3"] > 0,
        "stress_sharpe_at_least_0p90": s["P3"]["net_sharpe"] >= .90,
        "stress_excess_vs_P2_positive": s["P3"]["cumulative_net_return"] > s["P2"]["cumulative_net_return"],
        "single_stress_contribution_at_most_30pct": concentration["largest_positive_episode_share"] <= .30,
    }
    return {"G2": {"passed": all(g2.values()), "checks": g2},
            "G3_ECONOMIC": {"passed": all(g3.values()), "checks": g3},
            "era_excess_P3_minus_P2": eras, "stress_concentration": concentration}


def run_history(root: Path) -> dict:
    cfg, manifest = require_historical_gate(root)
    claim = {"MODEL_ID": MODEL_ID, "claimed_at": now(), "commit": git(root, "rev-parse", "HEAD"),
             "manifest_sha256": sha256(root / MANIFEST), "rerun_allowed": False}
    write_new(root / f"{REPORT}/historical_one_shot_claim.json", claim)
    # 第一处读取市场价格；前面的冻结、登记、G0/G1均不经过本函数。
    prices, dividends = load_market(root, cfg)
    predictions = pd.read_parquet(root / f"{DATA}/B0_B1_policy_targets.parquet")
    primary_schedule = make_schedule(predictions, prices, 0)
    primary = evaluate_portfolios(prices, dividends, primary_schedule, cfg)
    for cost, group in primary["ledgers"].items():
        for portfolio, ledger in group.items():
            ledger.to_parquet(root / f"{DATA}/{portfolio}_{cost}_daily_ledger.parquet", index=False)
    gates = economic_gates(primary, cfg)
    offsets = {}
    for offset in range(5):
        evaluated = primary if offset == 0 else evaluate_portfolios(
            prices, dividends, make_schedule(predictions, prices, offset), cfg)
        b = evaluated["metrics"]["base"]
        offsets[str(offset)] = {"P3_minus_P2_cumulative_excess":
            b["P3"]["cumulative_net_return"] - b["P2"]["cumulative_net_return"],
            "P3_net_sharpe": b["P3"]["net_sharpe"], "P2_net_sharpe": b["P2"]["net_sharpe"]}
    boot = bootstrap_lower(primary["ledgers"]["base"]["P3"], cfg)
    dsr = deflated_diagnostic(primary["ledgers"]["base"]["P3"], manifest)
    write_new(root / f"{REPORT}/economic_and_bootstrap_stage.json", {
        "gates": gates, "metrics": primary["metrics"], "offsets": offsets, "bootstrap": boot,
        "dsr": dsr, "state": "ONE_SHOT_RUNNING_PLACEBO_PENDING"})
    placebo_result = placebo(prices, dividends, primary_schedule,
        primary["ledgers"]["base"]["P3"], primary["ledgers"]["base"]["P1"], cfg)
    stat_checks = {"bootstrap_sharpe_lower_above_0p50": boot["one_sided_90pct_lower"] > .50,
        "four_positive_offsets": sum(x["P3_minus_P2_cumulative_excess"] > 0 for x in offsets.values()) >= 4,
        "placebo_top_10pct": placebo_result["passed"], "formal_deflated_sharpe_pass": dsr["passed"]}
    economic_pass = gates["G2"]["passed"] and gates["G3_ECONOMIC"]["passed"]
    if not economic_pass:
        state = "REJECTED_FROZEN_NO_RESCUE"
    elif not dsr["passed"]:
        state = "BLOCKED_STATISTICAL_ADMISSION_DISCOVERY_ONLY"
    elif not all(stat_checks.values()):
        state = "REJECTED_FROZEN_NO_RESCUE"
    else:
        state = cfg["adjudication"]["historical_pass"]
    result = {"MODEL_ID": MODEL_ID, "completed_at": now(), "STATE": state,
        "FAMILY_ID": cfg["adjudication"]["family_id"], "metrics": primary["metrics"],
        "P1_fixed_ex_post_target": primary["P1_fixed_ex_post_target"], "gates": gates,
        "statistics": {"checks": stat_checks, "bootstrap": boot, "deflated_sharpe": dsr, "placebo": placebo_result},
        "offsets": offsets, "primary_offset": 0, "PORTFOLIO_RETURN_READS": "ONE_SHOT_CONSUMED",
        "EVIDENCE_CLASS": cfg["evidence_class"], "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False,
        "CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY": "NONE"}
    write_new(root / f"{REPORT}/historical_result.json", result)
    write_new(root / f"{REPORT}/status.json", result)
    return result


def forward_calendar(root: Path, cfg: dict) -> pd.DatetimeIndex:
    historical = pd.read_csv(root / cfg["inputs"]["calendar"], usecols=["trade_date"])
    official = pd.read_csv(root / cfg["inputs"]["calendar_2026"], usecols=["trade_date"])
    old = pd.DatetimeIndex(pd.to_datetime(historical.trade_date))
    new = pd.DatetimeIndex(pd.to_datetime(official.trade_date))
    if not old[old.year == 2026].equals(new):
        raise ContractError("2026交易日历与官方日历不一致")
    if old.has_duplicates or not old.is_monotonic_increasing:
        raise ContractError("交易日历重复或不递增")
    return old


def register_forward(root: Path) -> dict:
    cfg, manifest = verify(root)
    stamp = pd.Timestamp(now())
    calendar = forward_calendar(root, cfg)
    anchor = pd.Timestamp(cfg["schedule"]["anchor_date"])
    index = calendar.get_indexer([anchor])[0]
    if index < 0:
        raise ContractError("父网格锚点不在日历中")
    future = calendar[index::5]
    future = future[future > stamp.tz_localize(None).normalize()]
    if len(future) == 0:
        raise ContractError("日历没有冻结之后的新原点")
    first = future[0]
    next_open = calendar[calendar > first][0]
    value = {"MODEL_ID": MODEL_ID, "registered_at": stamp.isoformat(),
        "manifest_sha256": sha256(root / MANIFEST), "freeze_commit": git(root, "rev-parse", "HEAD"),
        "STATE": "REGISTERED_WAITING_FIRST_NEW_ORIGIN",
        "first_new_origin_date": first.strftime("%Y-%m-%d"), "first_possible_execution_date": next_open.strftime("%Y-%m-%d"),
        "calendar_coverage_end": calendar.max().strftime("%Y-%m-%d"),
        "forward_origin_count": 0, "valid_forecast_count": 0, "shadow_position_target": None,
        "producer_state": "NO_VIEW_MISSING_STRICT_FORWARD_B0_B1_VINTAGE",
        "historical_prediction_not_forward": True, "backfill_allowed": False,
        "PORTFOLIO_RETURN_READS": 0, "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False}
    write_new(root / f"{FORWARD}/registration.json", value)
    return value


def shadow_tick(root: Path) -> dict:
    """追加严格前向运行状态；无合格供应凭据时只记NO_VIEW，不制造预测。"""
    cfg, manifest = verify(root)
    reg = read_json(root / f"{FORWARD}/registration.json")
    current = pd.Timestamp(now())
    today = current.tz_localize(None).normalize()
    status_path = root / f"{REPORT}/status.json"
    terminal = read_json(status_path).get("STATE") if status_path.exists() else None
    calendar = forward_calendar(root, cfg)
    index = calendar.get_indexer([pd.Timestamp(cfg["schedule"]["anchor_date"])])[0]
    grid = calendar[index::5]
    due = grid[(grid >= pd.Timestamp(reg["first_new_origin_date"])) & (grid <= today)]
    blocked = terminal in {"NO_VIEW_INSUFFICIENT_POLICY_VARIATION", "REJECTED_FROZEN_NO_RESCUE"}
    state = "STATUS_ONLY_FROZEN_POLICY" if blocked else (
        "WAITING_FIRST_NEW_ORIGIN" if len(due) == 0 else "NO_VIEW_MISSING_STRICT_FORWARD_B0_B1_VINTAGE")
    origin_dir = root / FORWARD / "origins"
    origin_dir.mkdir(parents=True, exist_ok=True)
    for origin in due:
        path = origin_dir / f"{origin.strftime('%Y%m%d')}.json"
        if path.exists() or (origin == today and current.hour < 15):
            continue
        reason = "STATUS_ONLY_FROZEN_POLICY" if blocked else (
            "MISSED_ORIGIN_WINDOW_NO_BACKFILL" if origin < today else "NO_VIEW_MISSING_STRICT_FORWARD_B0_B1_VINTAGE")
        write_new(path, {"MODEL_ID": MODEL_ID, "origin_date": origin.strftime("%Y-%m-%d"),
            "recorded_at": current.isoformat(), "state": reason, "valid_prediction": False,
            "shadow_position_target": None, "forecast_forward_filled": False,
            "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False})
    value = {"MODEL_ID": MODEL_ID, "recorded_at": current.isoformat(), "STATE": state,
        "historical_terminal_state": terminal, "scheduled_origins_elapsed": len(due),
        "valid_forward_origins": 0, "shadow_position_target": None,
        "next_origin": next((d.strftime("%Y-%m-%d") for d in grid if d > today), None),
        "52_origin_review": "NOT_MATURE", "104_origin_review": "NOT_MATURE", "156_origin_review": "NOT_MATURE",
        "POSITION_IMPACT": 0, "LIVE_TRADING_AUTHORIZED": False, "PORTFOLIO_RETURN_READS": 0}
    name = current.strftime("%Y%m%dT%H%M%S%f")
    write_new(root / FORWARD / "checks" / f"{name}.json", value)
    return value
