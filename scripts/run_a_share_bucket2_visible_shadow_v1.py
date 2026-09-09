"""采集桶2增量行情并刷新可见Shadow信号与模拟账户。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_bucket2_visible_shadow_v1 import (  # noqa: E402
    VisibleShadowError,
    extend_total_return_history,
    is_scheduled_signal_date,
    select_signal,
    target_frame,
)
from research.small_account_cross_sectional import (  # noqa: E402
    SmallAccountCosts,
    run_small_account_open_backtest,
)
from scripts.download_a_share_hash_holdout_training_v1 import (  # noqa: E402
    fetch_master,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.download_h00300_total_return import normalize_columns  # noqa: E402


CONFIG_PATH = ROOT / "config" / "a_share_bucket2_visible_shadow_v1.yaml"


class NonTradingDay(RuntimeError):
    """目标日期不是SSE交易日。"""


class TushareFailoverApi:
    """在已审核Tushare节点之间对每次API调用执行故障转移。"""

    def __init__(self, clients: list[tuple[str, Any]]) -> None:
        if not clients:
            raise ValueError("Tushare故障转移客户端不能为空")
        self._clients = clients
        self._active_index = 0
        self._successful_endpoints: list[str] = []
        self._failure_events: list[dict[str, str]] = []

    @property
    def endpoint(self) -> str:
        return self._clients[self._active_index][0]

    @property
    def successful_endpoints(self) -> list[str]:
        return list(self._successful_endpoints)

    @property
    def failure_events(self) -> list[dict[str, str]]:
        return list(self._failure_events)

    def __getattr__(self, method_name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            order = [self._active_index] + [
                index for index in range(len(self._clients)) if index != self._active_index
            ]
            errors: list[str] = []
            for index in order:
                endpoint, client = self._clients[index]
                try:
                    value = getattr(client, method_name)(*args, **kwargs)
                    self._active_index = index
                    if endpoint not in self._successful_endpoints:
                        self._successful_endpoints.append(endpoint)
                    return value
                except Exception as exc:
                    self._failure_events.append(
                        {
                            "method": method_name,
                            "endpoint": endpoint,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:160],
                        }
                    )
                    errors.append(f"{endpoint}: {type(exc).__name__}: {str(exc)[:160]}")
            raise RuntimeError(f"Tushare调用{method_name}时所有已审核节点均失败：{errors}")

        return call


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层不是对象：{path}")
    return value


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Shadow配置顶层不是对象")
    return value


def _validate_safety(config: dict[str, Any]) -> None:
    governance = config["governance"]
    if governance["visible_shadow_target_generation_enabled"] is not True:
        raise ValueError("可见Shadow目标生成未授权")
    if governance["simulated_account_mapping_enabled"] is not True:
        raise ValueError("模拟账户映射未授权")
    for field in (
        "real_position_mapping_enabled",
        "order_file_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        if governance[field] is not False:
            raise ValueError(f"真实交易安全开关未关闭：{field}")


def _verify_parent_freeze(config: dict[str, Any]) -> None:
    evidence_path = _inside(config["inputs"]["frozen_parent_evidence"])
    expected_manifest_hash = config["inputs"]["frozen_parent_evidence_sha256"]
    if sha256(evidence_path) != expected_manifest_hash:
        raise ValueError("旧桶2冻结证据清单哈希变化")
    evidence = _json(evidence_path)
    warmup_relative = config["inputs"]["frozen_warmup_panel"]
    item = next(
        (entry for entry in evidence["evidence"] if entry["path"] == warmup_relative),
        None,
    )
    if item is None:
        raise ValueError("旧桶2冻结包未登记暖启动面板")
    warmup_path = _inside(warmup_relative)
    if warmup_path.stat().st_size != item["bytes"] or sha256(warmup_path) != item["sha256"]:
        raise ValueError("旧桶2暖启动面板发生变化")


def _verify_implementation_manifest(config: dict[str, Any]) -> None:
    manifest_path = _inside(config["paths"]["protocol_manifest"])
    if not manifest_path.exists():
        raise ValueError("可见Shadow实施清单不存在")
    manifest = _json(manifest_path)
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ValueError("可见Shadow实施清单项目ID不一致")
    if manifest.get("strict_blind_claim_allowed") is not False:
        raise ValueError("实施清单错误地允许严格盲测结论")
    frozen_files = manifest.get("frozen_implementation_files")
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("实施清单没有冻结实现文件")
    mismatches: list[str] = []
    for relative, expected in frozen_files.items():
        path = _inside(str(relative))
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches.append(f"{relative}: expected={expected}, actual={actual}")
    if mismatches:
        raise ValueError(f"可见Shadow冻结实现发生变化：{mismatches}")


def _connect_api(config: dict[str, Any], probe_date: pd.Timestamp) -> tuple[TushareFailoverApi, str]:
    secret, configured = credentials()
    ts.set_token(secret)
    endpoints = list(dict.fromkeys([configured, "https://tt.xiaodefa.cn", "https://fast.xiaodefa.cn"]))
    clients: list[tuple[str, Any]] = []
    for endpoint in endpoints:
        api = ts.pro_api()
        api._DataApi__http_url = endpoint
        clients.append((endpoint, api))
    failover = TushareFailoverApi(clients)
    probe = failover.trade_cal(
        exchange="SSE",
        start_date=probe_date.strftime("%Y%m%d"),
        end_date=probe_date.strftime("%Y%m%d"),
        fields="exchange,cal_date,is_open",
    )
    if probe is None or probe.empty:
        raise RuntimeError("已审核Tushare节点交易日历探测返回空表")
    return failover, failover.endpoint


def _fetch_calendar(api: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    value = api.trade_cal(
        exchange="SSE",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        fields="exchange,cal_date,is_open",
    )
    if value is None or value.empty:
        raise RuntimeError("SSE交易日历返回空表")
    value["cal_date"] = pd.to_datetime(value["cal_date"], format="%Y%m%d", errors="coerce")
    dates = value.loc[value["is_open"].astype(int).eq(1), "cal_date"].dropna().sort_values()
    return pd.DatetimeIndex(dates)


def _fetch_master_snapshot(api: Any, config: dict[str, Any], as_of: pd.Timestamp) -> pd.DataFrame:
    adapter = {
        "universe": {
            "stock_statuses": config["data"]["stock_statuses"],
            "exchanges": config["data"]["exchanges"],
        },
        "split": {"holdout_remainder": config["split"]["fixed_bucket"]},
    }
    master = fetch_master(api, adapter)
    master["snapshot_date"] = as_of.normalize()
    return master


def _normalize_daily(value: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if value is None or value.empty:
        raise RuntimeError("Tushare全市场日线返回空表")
    frame = value.rename(
        columns={
            "ts_code": "con_code",
            "trade_date": "date",
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
        }
    ).copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    numeric = [
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "vol",
        "amount",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(
        subset=["date", "con_code", "pre_close", "raw_open", "raw_high", "raw_low", "raw_close"]
    )
    if frame[["pre_close", "raw_open", "raw_high", "raw_low", "raw_close"]].le(0).any().any():
        raise ValueError("增量行情存在非正价格")
    frame["volume"] = frame["vol"].fillna(0.0) * float(config["data"]["volume_multiplier_to_shares"])
    frame["amount"] = frame["amount"].fillna(0.0) * float(config["data"]["amount_multiplier_to_cny"])
    return frame.sort_values(["con_code", "date"]).reset_index(drop=True)


def _collect_daily_checkpoints(
    api: Any,
    dates: pd.DatetimeIndex,
    master: pd.DataFrame,
    config: dict[str, Any],
) -> list[Path]:
    directory = _inside(config["paths"]["daily_raw"])
    directory.mkdir(parents=True, exist_ok=True)
    bucket_codes = set(
        master.loc[
            master["split_bucket"].eq(int(config["split"]["fixed_bucket"])),
            "ts_code",
        ].astype(str)
    )
    paths: list[Path] = []
    fields = "ts_code,trade_date,pre_close,open,high,low,close,vol,amount"
    for trade_date in dates:
        path = directory / f"{trade_date:%Y%m%d}.parquet"
        if not path.exists():
            raw = api.daily(trade_date=trade_date.strftime("%Y%m%d"), fields=fields)
            frame = _normalize_daily(raw, config)
            frame = frame.loc[frame["con_code"].astype(str).isin(bucket_codes)].copy()
            if frame.empty:
                raise RuntimeError(f"{trade_date.date()}桶2日线为空")
            _atomic_parquet(frame, path)
        paths.append(path)
    return paths


def _download_benchmark(start: pd.Timestamp, end: pd.Timestamp, path: Path) -> pd.DataFrame:
    value = ak.stock_zh_index_hist_csindex(
        symbol="H00300",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    frame = normalize_columns(value)
    required = {"date", "symbol", "name", "close"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"H00300缺少字段：{sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.loc[
        frame["date"].between(start, end),
        ["date", "symbol", "name", "close"],
    ].dropna().drop_duplicates("date").sort_values("date")
    if frame.empty or frame["date"].max() < end:
        raise RuntimeError(f"H00300未覆盖目标日期{end.date()}")
    frame["source"] = "akshare.stock_zh_index_hist_csindex"
    _atomic_parquet(frame, path)
    return frame


def _costs(config: dict[str, Any]) -> SmallAccountCosts:
    costs = config["costs"]
    account = config["account"]
    return SmallAccountCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(costs["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(costs["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(costs["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(costs["slippage_bps_per_leg"]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
        minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


def _render(state: dict[str, Any]) -> str:
    target = state["current_target"]
    account = state["shadow_account"]
    lines = [
        "# 桶2固定低波公式可见Shadow",
        "",
        f"- 状态：`{state['status']}`",
        f"- 数据交易日：`{state['data_trade_date']}`",
        f"- 当前信号日：`{target['signal_date']}`",
        f"- Shadow目标：`{target['con_code']}` {target.get('name') or ''}",
        f"- 20日波动率：{target['raw_vol20']:.6f}",
        f"- 候选数量：{target['eligible_count']}",
        f"- 下一次换仓剩余：{state['trading_days_until_next_signal']}个交易日",
        "",
        "## 模拟账户",
        "",
        f"- 净值：{account['equity_cny']:.2f}元",
        f"- 现金：{account['cash_cny']:.2f}元",
        f"- 持仓数：{account['position_count']}",
        f"- 累计收益：{account['cumulative_return']:.2%}",
        f"- H00300同期收益：{account['benchmark_cumulative_return']:.2%}",
        f"- 状态说明：{state['action_message']}",
        "",
        "## 证据边界",
        "",
        f"- 等级：`{state['evidence_grade']}`",
        f"- 完整60日周期：{state['completed_cycles']}/10（首次统计摘要门槛）",
        f"- 全局治理：`{state['global_governance_status']}`",
        "- 这是可见Shadow模拟，不是盲测，不生成真实订单，也不连接券商。",
        "",
    ]
    return "\n".join(lines)


def _receipt_path(config: dict[str, Any], now: datetime) -> Path:
    directory = _inside(config["paths"]["run_receipts"])
    return directory / f"{now:%Y%m%dT%H%M%S%f}.json"


def _retained_state_report(
    payload: dict[str, Any],
    prior_state: dict[str, Any] | None,
    note: str,
) -> str:
    lines = [
        "# 桶2可见Shadow当前状态",
        "",
        f"- 本次运行：`{payload['run_status']}`",
        f"- 采集状态：`{payload['collection_status']}`",
        f"- 请求日期：`{payload['as_of_date']}`",
        "- 旧状态冒充本次成功：`false`",
    ]
    if payload.get("error_type"):
        lines.extend(
            [
                f"- 失败类型：`{payload['error_type']}`",
                f"- 失败原因：{payload['error']}",
            ]
        )
    if prior_state is not None:
        target = prior_state.get("current_target") or {}
        lines.extend(
            [
                "",
                "## 仅作留痕的上次成功状态",
                "",
                f"- 数据日期：`{prior_state.get('data_trade_date')}`",
                f"- 状态：`{prior_state.get('status')}`",
                f"- 目标：`{target.get('con_code')}` {target.get('name') or ''}",
                f"- 注意：{note}",
            ]
        )
    lines.extend(
        [
            "",
            "## 安全边界",
            "",
            "- 严格盲测结论：`false`",
            "- 实盘仓位映射：`false`",
            "- 订单生成：`false`",
            "- 券商连接：`false`",
            "- 实盘交易：`false`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_failure(config: dict[str, Any], as_of: pd.Timestamp, now: datetime, exc: Exception) -> None:
    state_path = _inside(config["paths"]["state"])
    prior_state = _json(state_path) if state_path.exists() else None
    payload = {
        "project_id": config["protocol"]["project_id"],
        "run_status": "FAILED",
        "collection_status": "NO_VIEW",
        "as_of_date": as_of.date().isoformat(),
        "generated_at": now.isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "stale_state_used_as_current": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    if prior_state is not None:
        payload["last_successful_data_trade_date"] = prior_state.get("data_trade_date")
        payload["last_successful_state_status"] = prior_state.get("status")
        payload["last_successful_target"] = prior_state.get("current_target")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_text(_inside(config["paths"]["last_run_status"]), content)
    _atomic_text(
        _inside(config["paths"]["current_report"]),
        _retained_state_report(payload, prior_state, "这不是本次新信号，必须等待下一次成功刷新。"),
    )
    _atomic_text(_receipt_path(config, now), content)


def _write_skip(config: dict[str, Any], as_of: pd.Timestamp, now: datetime) -> None:
    state_path = _inside(config["paths"]["state"])
    prior_state = _json(state_path) if state_path.exists() else None
    payload = {
        "project_id": config["protocol"]["project_id"],
        "run_status": "SKIPPED",
        "collection_status": "NON_TRADING_DAY",
        "as_of_date": as_of.date().isoformat(),
        "generated_at": now.isoformat(),
        "stale_state_used_as_current": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    if prior_state is not None:
        payload["last_successful_data_trade_date"] = prior_state.get("data_trade_date")
        payload["last_successful_state_status"] = prior_state.get("status")
        payload["last_successful_target"] = prior_state.get("current_target")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_text(_inside(config["paths"]["last_run_status"]), content)
    _atomic_text(
        _inside(config["paths"]["current_report"]),
        _retained_state_report(payload, prior_state, "非交易日没有生成新信号。"),
    )
    _atomic_text(_receipt_path(config, now), content)


def run(as_of: pd.Timestamp) -> dict[str, Any]:
    config = _config()
    _validate_safety(config)
    _verify_implementation_manifest(config)
    _verify_parent_freeze(config)
    now = datetime.now(ZoneInfo(config["data"]["timezone"]))
    governance_path = ROOT / "reports" / "audit" / "research_status_current.json"
    governance_status = _json(governance_path).get("overall_status", "UNKNOWN")
    warmup_path = _inside(config["inputs"]["frozen_warmup_panel"])
    warmup = pd.read_parquet(warmup_path)
    warmup["date"] = pd.to_datetime(warmup["date"])
    last_warmup_date = pd.Timestamp(warmup["date"].max())
    api, _ = _connect_api(config, as_of)
    if _fetch_calendar(api, as_of, as_of).empty:
        raise NonTradingDay(f"{as_of.date()}不是SSE交易日")
    master = _fetch_master_snapshot(api, config, as_of)
    snapshot_path = _inside(config["paths"]["master_snapshots"]) / f"{now:%Y%m%dT%H%M%S%f}.parquet"
    _atomic_parquet(master, snapshot_path)
    _atomic_parquet(master, _inside(config["paths"]["current_master"]))
    open_dates = _fetch_calendar(api, last_warmup_date + pd.Timedelta(days=1), as_of)
    if open_dates.empty:
        raise RuntimeError(f"{last_warmup_date.date()}之后没有可用交易日")
    checkpoint_paths = _collect_daily_checkpoints(api, open_dates, master, config)
    raw_increment = pd.concat([pd.read_parquet(path) for path in checkpoint_paths], ignore_index=True)
    panel = extend_total_return_history(
        warmup,
        raw_increment,
        float(config["data"]["return_identity_tolerance"]),
    )
    latest_trade_date = min(pd.Timestamp(panel["date"].max()), pd.Timestamp(open_dates.max()))
    if latest_trade_date < as_of.normalize():
        raise RuntimeError(f"行情仅到{latest_trade_date.date()}，不能冒充{as_of.date()}")
    _atomic_parquet(panel, _inside(config["paths"]["extended_panel"]))
    benchmark = _download_benchmark(
        pd.Timestamp(panel["date"].min()),
        latest_trade_date,
        _inside(config["paths"]["benchmark"]),
    )
    calendar = pd.DatetimeIndex(
        benchmark.loc[benchmark["date"].le(latest_trade_date), "date"].sort_values().unique()
    )
    state_path = _inside(config["paths"]["state"])
    prior_state = _json(state_path) if state_path.exists() else None
    first_signal_date = (
        pd.Timestamp(prior_state["first_signal_date"])
        if prior_state is not None
        else latest_trade_date
    )
    targets_path = _inside(config["paths"]["targets"])
    targets = pd.read_parquet(targets_path) if targets_path.exists() else pd.DataFrame()
    if not targets.empty and "eligible_count" not in targets.columns:
        if prior_state is None:
            raise VisibleShadowError("旧目标缺少候选数量且没有可核对状态")
        targets["eligible_count"] = int(prior_state["current_target"]["eligible_count"])
        _atomic_parquet(targets, targets_path)
    scheduled = is_scheduled_signal_date(
        calendar,
        first_signal_date,
        latest_trade_date,
        int(config["periods"]["rebalance_every_trading_days"]),
    )
    existing_signal_dates = (
        set(pd.to_datetime(targets["signal_date"])) if not targets.empty else set()
    )
    latest_selection = None
    if scheduled and latest_trade_date not in existing_signal_dates:
        latest_selection, _, _ = select_signal(panel, master, benchmark, latest_trade_date, config)
        new_target = target_frame(latest_selection)
        targets = new_target if targets.empty else pd.concat([targets, new_target], ignore_index=True)
        targets.sort_values("signal_date", inplace=True)
        _atomic_parquet(targets, targets_path)
    if targets.empty:
        raise VisibleShadowError("计划信号日没有生成Shadow目标")
    targets["signal_date"] = pd.to_datetime(targets["signal_date"])
    execution_calendar = calendar[(calendar >= first_signal_date) & (calendar <= latest_trade_date)]
    execution = panel[
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]
    ].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    ledger, trades = run_small_account_open_backtest(
        execution,
        targets,
        execution_calendar,
        float(config["account"]["initial_cash_cny"]),
        _costs(config),
        float(config["universe"]["maximum_open_total_return_gap_for_trade"]),
    )
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "date", "signal_date", "con_code", "side", "raw_open",
                "estimated_quantity", "notional", "commission", "stamp_duty",
                "slippage", "regime",
            ]
        )
    _atomic_parquet(ledger, _inside(config["paths"]["ledger"]))
    _atomic_parquet(trades, _inside(config["paths"]["trades"]))
    latest_target = targets.sort_values("signal_date").iloc[-1]
    frozen_master = pd.read_parquet(_inside(config["inputs"]["frozen_parent_master"]))
    master_lookup = frozen_master.drop_duplicates("ts_code").set_index("ts_code")
    target_name = (
        str(master_lookup.loc[str(latest_target["con_code"]), "name"])
        if str(latest_target["con_code"]) in master_lookup.index
        else None
    )
    latest_ledger = ledger.iloc[-1]
    benchmark_start = float(benchmark.loc[benchmark["date"].eq(first_signal_date), "close"].iloc[0])
    benchmark_end = float(benchmark.loc[benchmark["date"].eq(latest_trade_date), "close"].iloc[0])
    initial_cash = float(config["account"]["initial_cash_cny"])
    benchmark_return = benchmark_end / benchmark_start - 1.0
    cumulative_return = float(latest_ledger["equity"]) / initial_cash - 1.0
    completed_cycles = max((len(execution_calendar) - 1) // int(config["periods"]["rebalance_every_trading_days"]), 0)
    days_into_cycle = max((len(execution_calendar) - 1) % int(config["periods"]["rebalance_every_trading_days"]), 0)
    days_until_next = int(config["periods"]["rebalance_every_trading_days"]) - days_into_cycle
    if latest_target["signal_date"] == latest_trade_date:
        status = "PENDING_T1_OPEN"
        action_message = "信号已生成；等待下一交易日开盘进行Shadow模拟成交。"
    elif int(latest_ledger["position_count"]) > 0:
        status = "HOLDING"
        action_message = "Shadow持仓按收盘总收益价格估值；不是实盘仓位。"
    elif int(latest_ledger["blocked_buy_count"]) > 0:
        status = "BLOCKED_BUY"
        action_message = "本次Shadow买入被固定执行约束阻止。"
    else:
        status = "CASH_WAIT_NEXT_REBALANCE"
        action_message = "Shadow账户保持现金，等待下一固定信号日。"
    current_target = {
        "signal_date": pd.Timestamp(latest_target["signal_date"]).date().isoformat(),
        "con_code": str(latest_target["con_code"]),
        "name": target_name,
        "score": float(latest_target["score"]),
        "raw_vol20": float(latest_target["raw_vol20"]),
        "eligible_count": int(latest_target["eligible_count"]),
    }
    state = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": now.isoformat(),
        "run_status": "SUCCESS",
        "status": status,
        "evidence_grade": config["protocol"]["evidence_grade"],
        "data_trade_date": latest_trade_date.date().isoformat(),
        "first_signal_date": first_signal_date.date().isoformat(),
        "current_target": current_target,
        "shadow_account": {
            "initial_cash_cny": initial_cash,
            "equity_cny": float(latest_ledger["equity"]),
            "cash_cny": float(latest_ledger["cash"]),
            "position_count": int(latest_ledger["position_count"]),
            "cumulative_return": cumulative_return,
            "benchmark_cumulative_return": benchmark_return,
            "cumulative_excess": cumulative_return - benchmark_return,
        },
        "completed_cycles": int(completed_cycles),
        "minimum_cycles_for_statistical_summary": int(config["periods"]["minimum_complete_cycles_for_statistical_summary"]),
        "statistical_status": "OBSERVATION_ONLY" if completed_cycles < int(config["periods"]["minimum_complete_cycles_for_statistical_summary"]) else "ELIGIBLE_FOR_PREREGISTERED_SUMMARY",
        "trading_days_until_next_signal": int(days_until_next),
        "action_message": action_message,
        "global_governance_status": governance_status,
        "global_governance_warning": governance_status != "PASS",
        "source_audit": {
            "tushare_endpoint": api.endpoint,
            "tushare_successful_endpoints": api.successful_endpoints,
            "tushare_failover_events": api.failure_events,
            "master_snapshot_date": as_of.date().isoformat(),
            "daily_checkpoint_dates": [path.stem for path in checkpoint_paths],
            "benchmark_last_date": pd.Timestamp(benchmark["date"].max()).date().isoformat(),
            "parent_warmup_sha256": sha256(warmup_path),
            "extended_panel_sha256": sha256(_inside(config["paths"]["extended_panel"])),
            "benchmark_sha256": sha256(_inside(config["paths"]["benchmark"])),
            "targets_sha256": sha256(targets_path),
            "ledger_sha256": sha256(_inside(config["paths"]["ledger"])),
        },
        "safety": config["governance"],
    }
    state_content = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    _atomic_text(state_path, state_content)
    _atomic_text(_inside(config["paths"]["last_run_status"]), state_content)
    _atomic_text(_inside(config["paths"]["current_report"]), _render(state))
    _atomic_text(_receipt_path(config, now), state_content)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="刷新桶2固定低波公式可见Shadow")
    parser.add_argument("--as-of", type=str, help="目标交易日，默认上海时区今天")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    config = _config()
    timezone = ZoneInfo(config["data"]["timezone"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(datetime.now(timezone).date())
    now = datetime.now(timezone)
    try:
        state = run(as_of.normalize())
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except NonTradingDay:
        _write_skip(config, as_of.normalize(), now)
        print(f"{as_of.date()}不是交易日，Shadow任务已正常跳过。")
        return 0
    except Exception as exc:
        _write_failure(config, as_of.normalize(), now, exc)
        print(f"可见Shadow刷新失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
