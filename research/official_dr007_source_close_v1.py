"""登记已取得的官方日终锚点，保留原预测及盘中、日终的区别。"""

import json
from decimal import Decimal

from company_revision_driver_sources_v1 import now, save
from official_dr007_source_v1 import OUT, ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮已登记，不重复运行。")
    report_name = "资金价格背后的原因与日终锚点.md"
    if not (OUT / report_name).exists():
        raise SystemExit("尚未保存本轮解释。")
    daily = read(OUT / "sources/official_dr_daily_20260928.json")
    receipt = read(OUT / "receipts/official_dr_daily_20260928.json")
    live = read(OUT / "sources/repo_live_data.json")
    baseline_path = ROOT / "reports/research/510300_domestic_event_baseline_v1"
    baseline = read(baseline_path / "reviewed_facts.json")
    previous = read(baseline_path / "result.json")
    cards = read(ROOT / "reports/research/510300_current_driver_outlook_20260929/forecast_cards.json")
    scope = read(OUT / "scope.json")
    actual_requests = len(list((OUT / "receipts").glob("*.json")))
    request_limit = scope["initial_requests"] + scope["additional_limit"]
    if actual_requests > request_limit:
        raise ValueError("实际请求数超过本轮范围。")
    if daily["head"]["rep_code"] != "200":
        raise ValueError("日行情业务响应未成功。")
    requested_date = receipt["payload"]["searchDate"]
    if daily["data"]["lastDate"] != requested_date:
        raise ValueError("日行情返回日期与请求不一致。")
    if receipt["payload"]["publishedTime"] != "2200":
        raise ValueError("请求未使用已核对的官方日报时点参数。")
    daily_rows = {row["instrmntCd"]: row for row in daily["records"]}
    live_rows = {row["productCode"]: row for row in live["records"]}
    policy = Decimal(str(baseline["funding"]["announcements"][requested_date.replace("-", "")]["seven_day_rate_percent"]))
    weighted = Decimal(daily_rows["DR007"]["wghtdAvgRepoRate"])
    closing = Decimal(daily_rows["DR007"]["clsngRepoRate"])
    dr001 = Decimal(daily_rows["DR001"]["wghtdAvgRepoRate"])
    dr014 = Decimal(daily_rows["DR014"]["wghtdAvgRepoRate"])
    spread = (weighted - policy) * Decimal("100")
    stamp = now()
    f3 = next(card for card in cards["cards"] if card["id"] == "F3")
    facts = {
        "recorded_at": stamp,
        "status": "OFFICIAL_PRIOR_DAY_WEIGHTED_RATE_ADMITTED_FORWARD_CARD_PENDING",
        "daily_observation": {
            "observation_date": requested_date,
            "received_at": receipt["received_at"],
            "source_response_end_date": daily["data"]["endDate"],
            "source": "https://www.chinamoney.com.cn/chinese/mtdexdaily/?tab=2",
            "business_response_code": daily["head"]["rep_code"],
            "dr007_weighted_rate_percent": float(weighted),
            "dr007_closing_rate_percent": float(closing),
            "policy_rate_percent": float(policy),
            "dr007_policy_spread_bp": float(spread),
            "dr001_weighted_rate_percent": float(dr001),
            "dr014_weighted_rate_percent": float(dr014),
            "dr007_minus_dr001_bp": float((weighted - dr001) * 100),
            "dr014_minus_dr007_bp": float((dr014 - weighted) * 100),
            "dr007_average_term_days": float(daily_rows["DR007"]["avgPrd"]),
            "dr014_average_term_days": float(daily_rows["DR014"]["avgPrd"]),
            "dr007_transactions": int(daily_rows["DR007"]["dlNo"]),
            "dr007_volume_100m_cny": float(daily_rows["DR007"]["trdVol"]),
            "publication_basis": "官方质押式回购日报以publishedTime=2200查询，业务成功，日期匹配；2200是页面参数，不是对未来准时发布的保证。",
            "revision_policy": "本次收到的官方日终版本；未来若更正，另记更正，不覆盖原始响应。",
            "f3_input": requested_date in f3["before_dates"] + f3["after_dates"],
        },
        "intraday_observation": {
            "source_time": live["data"]["showDateCN"],
            "dr007_weighted_rate_percent": float(live_rows["DR007"]["weightedRate"]),
            "dr007_latest_rate_percent": float(live_rows["DR007"]["latestRate"]),
            "is_daily_final": False,
            "f3_input": False,
        },
        "official_request_recipe": {
            "method": "POST",
            "url": receipt["url"],
            "form_fields_as_observed": receipt["payload"],
            "date_field": "data.lastDate",
            "instrument_field": "records[].instrmntCd == DR007",
            "weighted_rate_field": "records[].wghtdAvgRepoRate",
            "do_not_use_as_weighted_rate": "records[].clsngRepoRate",
            "parameter_evidence": "sources/daily_repo_component.html及sources/official_common_script.html",
            "automated_fetch_created": False,
        },
        "causal_interpretation": {
            "supported": "9月28日银行间存款类机构7天回购加权价格接近当日政策利率；当日未见该指标明显高于政策利率。",
            "supply_hypothesis": "跨季供给安排和续作缓冲资金需求；当前观测与此相容，未识别其中因果份额。",
            "alternatives": [
                "融资或持仓需求偏弱也能使资金价格低，股票盈利含义可能相反。",
                "现金、财政和其他工具改变资金净供求，已知逆回购本金不覆盖全部渠道。",
                "期限与成交构成改变加权价格；DR007实际平均期限6.80天，并非同质精确7天合约。",
            ],
            "unidentified": ["银行与非银资金价差", "新增股票购买资金", "相对于事前市场预期的意外程度", "沪深300未来盈利和估值净效应"],
            "liquidity_to_equity_bridge": "融资约束缓解须继续传到可执行的信用或投资需求、经营现金流和风险承担；资金价格平稳本身不生成指数看涨信号。",
        },
        "forecast_state": [{"id": c["id"], "status": c["status"]} for c in cards["cards"]],
        "f3_missing_dates": f3["before_dates"] + f3["after_dates"],
        "prior_date_not_substituted": True,
        "new_accounts": 0,
        "new_return_tests": 0,
        "new_forecast_cards": 0,
        "new_fitted_parameters": 0,
        "direct_requests": actual_requests,
    }
    save(OUT / "reviewed_facts.json", facts)
    next_question = "按真实新发布数据评价既定预测：官方日报日期推进后保留F3指定日期，9月30日PMI先区分订单、交付与价格原因，再比较事前预期及指数价格。"
    result = {
        "study_id": "510300_OFFICIAL_DR007_SOURCE_V1",
        "recorded_at": stamp,
        "previous_goal_turn_classification": previous["continuation_classification"],
        "continuation_classification": "PROGRESS_OFFICIAL_DAILY_FUNDING_ANCHOR_AND_SOURCE_RESOLUTION",
        "status": facts["status"],
        "concrete_progress": [
            "从官方日报组件取得正确的indexType、2200时点和加权字段，解决缺参业务错误。",
            "取得9月28日官方日终DR007为1.3911%，对应政策利率差-0.89bp。",
            "区分供给缓冲、需求偏弱和成交构成三个解释，保持指数含义及原F3待评。",
        ],
        "new_accounts": 0,
        "new_strategy_return_tests": 0,
        "new_forecast_cards": 0,
        "new_fitted_parameters": 0,
        "existing_account_counts": previous["existing_account_counts"],
        "original_forecasts_unchanged": True,
        "historical_rejections_unchanged": True,
        "net_sharpe": None,
        "net_cagr": None,
        "performance_status": "NOT_COMPUTED",
        "goal_status": "active",
        "goal_achieved": False,
        "not_a_blocked_turn": True,
        "live_handle": None,
        "verified_wait": False,
        "orders_authorized": False,
        "report": report_name,
        "next_question": next_question,
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(path)
    protocol.update({
        "latest_concrete_diagnostic": rel + "/result.json",
        "latest_official_funding_observation": rel + "/reviewed_facts.json",
        "current_source_admission": rel + "/reviewed_facts.json",
        "updated_at": stamp,
    })
    save(path, protocol)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update({
        "current_round": result["study_id"],
        "latest_progress_receipt": rel + "/result.json",
        "latest_continuation_report": rel + "/" + report_name,
        "latest_continuation_classification": result["continuation_classification"],
        "last_research_result": "9月28日官方日终DR007为1.3911%，比当日政策利率低0.89bp；与资金供给缓冲相容，也不能排除需求偏弱。没有据此推定股票买盘或净收益优势。",
        "last_source_result": "官方组件已给出markInterBankVOList及publishedTime=2200，日行情业务成功且返回日期匹配9月28日。9月29日仅盘中值；F3全部指定日期仍待最终观测。",
        "latest_driver_diagnostic_at": stamp,
        "latest_official_funding_observation": rel + "/reviewed_facts.json",
        "next_research_question": next_question,
    })
    save(path, mandate)
    path = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- OFFICIAL_DR007_SOURCE_V1_20260929 -->

## 2026-09-29 官方日终资金锚点与传导边界

从官方页面取得日报参数及加权字段，解决前轮缺参错误。9月28日DR007为1.3911%，比当日7天政策利率低0.89bp；9月29日15:00的1.3862%保留为盘中值。所取28日不替换F3的29、30日基准，F1/F2/F3继续待评。供给缓冲、需求偏弱和成交构成是不同解释，资金平稳不能直接推出指数利好。新增账户与收益测试0，旧失败不改；目标active且未实现，本轮PROGRESS，无后台等待任务。

详见[资金价格背后的原因与日终锚点](<{(OUT / report_name).as_posix()}>)。

"""
    path.write_text(head + path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"已登记官方日终DR007 {weighted}%，政策利差{spread}基点；原预测仍待评，本轮请求{actual_requests}次。")


if __name__ == "__main__":
    main()
