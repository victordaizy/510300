"""510300 PCF、IOPV与2万元纸面执行测试。"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from backtest.small_account_execution import (
    build_small_account_paper_target,
    quantize_position,
)
from market_data.etf_primary_market import (
    EtfPrimaryMarketRequest,
    IOPV_SELECT_FIELDS,
    SseIopvSnapshotProvider,
    SsePcfProvider,
)
from research.primary_market_forward_readiness import evaluate_forward_readiness
from scripts.collect_510300_primary_market import _iso_calendar_date


class _FakeResponse:
    def __init__(self, *, text: str | None = None, payload: dict | None = None) -> None:
        self.text = text or ""
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        if self._payload is None:
            raise RuntimeError("测试响应没有JSON载荷")
        return self._payload


class _FakePcfSession:
    def get(self, url: str, params: dict, headers: dict, timeout: float) -> _FakeResponse:
        self.last_params = params
        result = {
            "NAV": "￥4.7527",
            "NET_REDEMPTION_LIMIT": "-",
            "REDEMPTION_LIMIT_PER_ACCT": "-",
            "CREATION_REDEMPTION_UNIT": "900000",
            "FUND_NAME": "华泰柏瑞沪深300交易型开放式指数证券投资基金",
            "FUND_COMP_NAME": "华泰柏瑞基金管理有限公司",
            "ETF_TYPE": "8",
            "CREATION_REDEMPTION": "申购和赎回皆允许",
            "NET_CREATION_LIMIT": "-",
            "NAVPERCU": "￥4277442.73",
            "CREATION_LIMIT": "-",
            "CREATION_LIMIT_PER_ACCT": "-",
            "TRADING_DAY": "20260813",
            "FILE_ID": "1461355",
            "REDEMPTION_LIMIT": "3600000000",
            "PRE_TRADING_DAY": "20260812",
            "PRE_CASH_COMPONENT": "￥107759.73",
            "ESTIMATED_CASH_COMPONENT": "￥103752.73",
            "RECORD_NUM": "300",
            "PUBLISH_IOPV": "是",
            "TRADE_CODE": "510300",
            "MAX_CASH_RATIO": "50%",
            "FUNDID1": "-",
            "CREATION_REDEMPTION_MECHANISM": "1",
            "NET_CREATION_LIMIT_PER_ACCT": "-",
            "NET_REDEMPTION_LIMIT_PER_ACCT": "-",
        }
        payload = {"result": [result]}
        return _FakeResponse(text=f"jsonpCallback({json.dumps(payload, ensure_ascii=False)})")


class _FakeIopvSession:
    def get(self, url: str, params: dict, headers: dict, timeout: float) -> _FakeResponse:
        self.last_params = params
        values = {
            "name": "300ETF",
            "last": 4.7290,
            "chg_rate": -0.40,
            "change": -0.0190,
            "open": 4.7700,
            "prev_close": 4.7480,
            "high": 4.7860,
            "low": 4.7250,
            "volume": 915582741,
            "amount": 4364799044,
            "tradephase": "E110    ",
            "cpxxextendname": "沪深300ETF华泰柏瑞",
            "iopv": 4.7266,
            "fp_volume": 592700,
            "fp_amount": 2802878.3,
            "fp_phase": "D0      ",
            "cpxxsubtype": "EBS",
        }
        payload = {
            "code": "510300",
            "date": 20260813,
            "time": 162906,
            "snap": [values[field] for field in IOPV_SELECT_FIELDS],
        }
        return _FakeResponse(payload=payload)


class PrimaryMarketProviderTests(unittest.TestCase):
    def test_PCF标准化且核对最小申购单位(self) -> None:
        session = _FakePcfSession()
        snapshot = SsePcfProvider(session=session).fetch(
            EtfPrimaryMarketRequest(fund_code="510300")
        )
        self.assertEqual(snapshot.record["creation_redemption_unit"], 900_000)
        self.assertEqual(snapshot.record["component_count"], 300)
        self.assertTrue(snapshot.record["publish_iopv"])
        self.assertAlmostEqual(snapshot.record["maximum_cash_ratio"], 0.5)
        self.assertEqual(session.last_params["FUNDID2"], "510300")

    def test_官方快照生成盘中折溢价(self) -> None:
        session = _FakeIopvSession()
        snapshot = SseIopvSnapshotProvider(session=session).fetch(
            EtfPrimaryMarketRequest(fund_code="510300")
        )
        self.assertEqual(str(snapshot.record["exchange_timestamp"]), "2026-08-13 16:29:06")
        self.assertEqual(snapshot.record["last_price"], 4.729)
        self.assertEqual(snapshot.record["iopv"], 4.7266)
        self.assertAlmostEqual(snapshot.record["premium_discount_bps"], 5.0776456650)
        self.assertIn("iopv", session.last_params["select"])


class SmallAccountExecutionTests(unittest.TestCase):
    def test_30仓位映射到25且目标1000份(self) -> None:
        target = build_small_account_paper_target(
            account_equity_cny=20_000,
            raw_target_position=0.30,
            price_cny=4.729,
            position_step=0.25,
            lot_size=100,
            minimum_normal_trade_shares=1000,
        )
        self.assertEqual(target["target_position_grid"], 0.25)
        self.assertEqual(target["target_shares"], 1000)
        self.assertEqual(target["paper_action"], "TARGET_ONLY_NO_HOLDINGS")
        self.assertFalse(target["automatic_ordering_authorized"])

    def test_小于1000份差额只累积不交易(self) -> None:
        target = build_small_account_paper_target(
            account_equity_cny=20_000,
            raw_target_position=0.30,
            price_cny=4.729,
            position_step=0.25,
            lot_size=100,
            minimum_normal_trade_shares=1000,
            current_shares=500,
            available_cash_cny=17_635.50,
            sellable_shares=500,
            today_bought_shares=0,
            allow_position_increase=True,
        )
        self.assertEqual(target["paper_trade_shares"], 0)
        self.assertEqual(target["paper_action"], "HOLD_ACCUMULATE_UNECONOMIC_DIFFERENCE")

    def test_满仓也会保留费用尾差(self) -> None:
        target = build_small_account_paper_target(
            account_equity_cny=20_000,
            raw_target_position=1.0,
            price_cny=4.729,
            position_step=0.25,
            lot_size=100,
            minimum_normal_trade_shares=1000,
            current_shares=0,
            available_cash_cny=20_000,
            sellable_shares=0,
            today_bought_shares=0,
            allow_position_increase=True,
        )
        self.assertEqual(quantize_position(1.0, 0.25), 1.0)
        self.assertEqual(target["target_shares"], 4200)
        self.assertEqual(target["paper_action"], "PAPER_BUY")

    def test_非复核日禁止把目标差额误报为交易(self) -> None:
        target = build_small_account_paper_target(
            account_equity_cny=20_000,
            raw_target_position=0.75,
            price_cny=5.0,
            position_step=0.25,
            lot_size=100,
            minimum_normal_trade_shares=1000,
            current_shares=1000,
            available_cash_cny=15_000,
            sellable_shares=1000,
            today_bought_shares=0,
            allow_position_increase=False,
            allow_position_decrease=False,
        )
        self.assertEqual(target["paper_action"], "HOLD_INCREASE_NOT_ALLOWED")
        self.assertEqual(target["paper_trade_shares"], 0)

    def test_T加1只允许卖出旧仓(self) -> None:
        target = build_small_account_paper_target(
            account_equity_cny=20_000,
            raw_target_position=0.0,
            price_cny=5.0,
            position_step=0.25,
            lot_size=100,
            minimum_normal_trade_shares=1000,
            current_shares=3000,
            available_cash_cny=5_000,
            sellable_shares=1000,
            today_bought_shares=2000,
            allow_position_decrease=True,
        )
        self.assertEqual(target["paper_action"], "PAPER_SELL")
        self.assertEqual(target["signed_paper_trade_shares"], -1000)


class CollectorDateTests(unittest.TestCase):
    def test_date对象与ISO字符串归一化后相等(self) -> None:
        self.assertEqual(_iso_calendar_date(pd.Timestamp("2026-08-14").date()), "2026-08-14")
        self.assertEqual(_iso_calendar_date("2026-08-14"), "2026-08-14")

    def test_计划任务使用前台研究采集且禁止纸面信号(self) -> None:
        root = Path(__file__).resolve().parents[1]
        installer = (root / "scripts" / "install_510300_daily_collection_task.ps1").read_text(
            encoding="utf-8"
        )
        task_runner = (
            root / "scripts" / "run_510300_primary_market_collection_task.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("run_510300_primary_market_collection_task.ps1", installer)
        self.assertIn("-WakeToRun", installer)
        self.assertIn("-RestartCount 3", installer)
        self.assertIn("-Watch -SkipPaperSignal", task_runner)
        self.assertIn("SKIPPED_NON_TRADING_DAY", task_runner)
        self.assertIn("FAILED_NO_CURRENT_TRADING_DAY_DATA", task_runner)
        self.assertIn("sse_trade_calendar_2026.csv", task_runner)
        self.assertIn("immutable_receipt = $true", task_runner)
        self.assertIn("primary_market_task_runs", task_runner)
        self.assertIn("live_trading_enabled = $false", task_runner)
        self.assertNotIn("generate_510300_small_account_paper_signal.py", task_runner)

    def test_遗留前台研究入口可由Windows_PowerShell_5解析(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = root / "scripts" / "run_510300_primary_market_forward.ps1"
        self.assertTrue(runner.read_bytes().startswith(b"\xef\xbb\xbf"))

        environment = os.environ.copy()
        environment["CODEX_POWERSHELL_PARSE_TARGET"] = str(runner)
        command = (
            "$tokens = $null; $errors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "$env:CODEX_POWERSHELL_PARSE_TARGET, [ref]$tokens, [ref]$errors) "
            "| Out-Null; if ($errors.Count -gt 0) { exit 1 }"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class ForwardReadinessTests(unittest.TestCase):
    def test_不足20个完整交易日保持采集中(self) -> None:
        pcf = pd.DataFrame(
            {
                "trading_day": ["2026-08-13"],
                "retrieved_at": ["2026-08-13T09:30:01+08:00"],
            }
        )
        iopv = pd.DataFrame(
            {
                "trade_date": ["2026-08-13", "2026-08-13"],
                "exchange_timestamp": [
                    "2026-08-13 09:31:00",
                    "2026-08-13 09:32:00",
                ],
                "retrieved_at": [
                    "2026-08-13T09:31:01+08:00",
                    "2026-08-13T09:32:01+08:00",
                ],
                "iopv": [4.72, 4.73],
            }
        )
        result = evaluate_forward_readiness(
            pcf,
            iopv,
            minimum_full_coverage_days=20,
            recommended_full_coverage_days=40,
            minimum_snapshots_per_day=120,
        )
        self.assertEqual(result["status"], "COLLECTING_NOT_ELIGIBLE")
        self.assertEqual(result["full_coverage_days"], 0)
        self.assertFalse(result["eligible_for_feature_freeze"])
        self.assertFalse(result["eligible_for_research_evaluation"])
        self.assertFalse(result["eligible_for_position_mapping"])

    def test_达到20日只允许质量审计不允许评价或仓位映射(self) -> None:
        dates = pd.date_range("2026-07-01", periods=20, freq="B")
        pcf = pd.DataFrame(
            {
                "trading_day": dates,
                "retrieved_at": [
                    f"{date.date()}T09:30:01+08:00" for date in dates
                ],
            }
        )
        rows = []
        for trade_date in dates:
            for minute in range(120):
                exchange_timestamp = trade_date + pd.Timedelta(
                    hours=9, minutes=30 + minute
                )
                rows.append(
                    {
                        "trade_date": trade_date,
                        "exchange_timestamp": exchange_timestamp,
                        "retrieved_at": exchange_timestamp.tz_localize(
                            "Asia/Shanghai"
                        ).isoformat(),
                        "iopv": 4.75,
                    }
                )
        result = evaluate_forward_readiness(
            pcf,
            pd.DataFrame(rows),
            minimum_full_coverage_days=20,
            recommended_full_coverage_days=40,
            minimum_snapshots_per_day=120,
        )
        self.assertEqual(
            result["status"], "QUALITY_AUDIT_ELIGIBLE_NOT_FEATURE_FREEZE"
        )
        self.assertTrue(result["eligible_for_quality_audit"])
        self.assertFalse(result["eligible_for_feature_freeze"])
        self.assertFalse(result["eligible_for_research_evaluation"])
        self.assertFalse(result["eligible_for_position_mapping"])

    def test_陈旧IOPV或无时区检索时间不能计为完整日(self) -> None:
        trade_date = pd.Timestamp("2026-08-18")
        pcf = pd.DataFrame(
            {
                "trading_day": [trade_date],
                "retrieved_at": ["2026-08-18T09:30:01+08:00"],
            }
        )
        rows = []
        for minute in range(120):
            exchange_timestamp = trade_date + pd.Timedelta(
                hours=9, minutes=30 + minute
            )
            rows.append(
                {
                    "trade_date": trade_date,
                    "exchange_timestamp": exchange_timestamp,
                    "retrieved_at": (
                        exchange_timestamp + pd.Timedelta(minutes=10)
                    ).isoformat(),
                    "iopv": 4.75,
                }
            )
        result = evaluate_forward_readiness(
            pcf,
            pd.DataFrame(rows),
            minimum_full_coverage_days=20,
            recommended_full_coverage_days=40,
            minimum_snapshots_per_day=120,
        )
        quality = result["daily_quality"][0]
        self.assertEqual(result["full_coverage_days"], 0)
        self.assertFalse(quality["checks"]["retrieval_timezone_valid"])
        self.assertFalse(quality["checks"]["iopv_staleness_within_limit"])

    def test_重复时间戳按交易日去重后不足120不计完整日(self) -> None:
        trade_date = pd.Timestamp("2026-08-18")
        pcf = pd.DataFrame(
            {
                "trading_day": [trade_date],
                "retrieved_at": ["2026-08-18T09:30:01+08:00"],
            }
        )
        timestamp = trade_date + pd.Timedelta(hours=9, minutes=30)
        iopv = pd.DataFrame(
            {
                "trade_date": [trade_date] * 120,
                "exchange_timestamp": [timestamp] * 120,
                "retrieved_at": ["2026-08-18T09:30:01+08:00"] * 120,
                "iopv": [4.75] * 120,
            }
        )
        result = evaluate_forward_readiness(
            pcf,
            iopv,
            minimum_full_coverage_days=20,
            recommended_full_coverage_days=40,
            minimum_snapshots_per_day=120,
        )
        quality = result["daily_quality"][0]
        self.assertEqual(quality["deduplicated_snapshot_count"], 1)
        self.assertEqual(quality["duplicate_snapshot_count"], 119)
        self.assertFalse(quality["complete_quality_day"])


if __name__ == "__main__":
    unittest.main()
