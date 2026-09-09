"""核对价格恐慌真实口径与完整账户，不重复回测。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / "scripts/finalize_round85_20260908.py"
    require(not dest.exists(), "85核对来源已存在")
    s = (ROOT / "scripts/finalize_round78_20260908.py").read_text(encoding="utf-8").split('    status = "COMPLETED_COMPOSITE_REVERSAL_TARGET_NOT_MET"')[0]
    s = s.replace("research.composite_streak_reversal_v1", "research.synthetic_fear_recovery_v1")
    s = s.replace("510300三项短期反转_第78轮_20260908", "510300价格恐慌回落_第85轮_20260908").replace("三项短期反转_结果及中文规则.md", "价格恐慌回落_结果及中文规则.md")
    s = s.replace("510300_AFTER_COMPOSITE_DIRECTIONAL_EXIT_20260908.md", "510300_AFTER_SYNTHETIC_FEAR_LIQUIDITY_EXIT_20260908.md")
    s = s.replace('== 77 and not OUT.exists()', '== 84 and not OUT.exists()').replace("78前序", "85前序")
    marker = '    dividends = normalize_dividends'
    place = s.index(marker)
    s = s[:place]+'''    data = pd.read_parquet(ROOT/cfg["features"])
    # 用前日财富与当日最低价重建低点尺度，独立于实现中的当日收盘比值。
    low = data.wealth.shift(1)*(data.low+data.dividend)/data.previous_close
    np.testing.assert_allclose(main_f.wealth_low.iloc[1:], low.iloc[1:], atol=1e-12, rtol=0, equal_nan=True)
    expected_high = data.wealth.rolling(22).max()
    expected_fear = 100*(expected_high-low)/expected_high
    expected_mean = expected_fear.rolling(20).mean()
    expected_sd = expected_fear.rolling(20).apply(lambda x: float(np.std(x, ddof=1)), raw=True)
    expected_upper = expected_mean+2*expected_sd
    for name, values in [("highest_close22", expected_high), ("fear22", expected_fear), ("fear_mean20", expected_mean),
        ("fear_sd20", expected_sd), ("fear_upper20", expected_upper)]:
        np.testing.assert_allclose(main_f[name], values, atol=1e-8, rtol=0, equal_nan=True)
    expected_entry = ((expected_fear.shift(1)>expected_upper.shift(1)) & (expected_fear<=expected_upper) & (data.wealth>data.wealth.shift(1))).fillna(False)
    expected_exit = data.wealth >= data.wealth.rolling(20).mean()
    np.testing.assert_array_equal(main_f.entry_condition, expected_entry.to_numpy(int))
    np.testing.assert_array_equal(main_f.price_exit_condition, expected_exit.to_numpy(bool))
''' + s[place:]
    s = s.replace('"KEY_CAUSAL_PREFIX_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE"', '"KEY_INDEPENDENT_FEAR_SCALE_CAUSAL_PREFIX_AND_COMPLETE_ACCOUNTS_CHECKED"')
    s = s.replace('"prefix_days": len(early_f),', '"prefix_days": len(early_f), "independent_factor_rows": len(main_f),')
    s += '''    economics = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_ledger.parquet")
            group = [c for c in cycles if c["period"] == period and c["cost"] == cost]
            economics.append({"period": period, "cost": cost, "cycles": len(group), "losing_cycles": sum(c["net_profit"]<0 for c in group),
                "price_and_dividend_before_explicit_cost": float((ledger.price_pnl+ledger.dividend_recognized).sum()),
                "explicit_cost": float((ledger.commission+ledger.slippage_cost).sum()), "net_profit": float(ledger.equity.iloc[-1]-cfg["initial_capital"])})
    pd.DataFrame(economics).to_csv(RESEARCH/"saved_signal_and_cost_attribution.csv", index=False, encoding="utf-8-sig")
    status = "COMPLETED_SYNTHETIC_FEAR_RECOVERY_TARGET_NOT_MET"
    write_json(RESEARCH/"acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "主夏普0.315、较早负0.251；较早扣除显式交易成本之前的价格与分红合计也亏损，不能只归因于费用。关闭指标回落策略，不改波动带、窗口和退出救回。"}, exclusive=True)
    NEXT.write_text("# 第85轮完成，下一项用现有成交额补充原退出模型\\n\\n"
        "85主基础／压力夏普0.315093／0.208147，较早负0.250901／负0.356702。主34周期68成交178收盘，较早23周期46成交86收盘，无未成交或评价期因子缺失。主基础年化1.8528%、回撤12.9735%；较早年化负1.7783%、回撤14.9316%。较早价格与分红扣显式费用前合计亏10533.70元，再支出6749.80元费用，净亏17283.50元；不是只因费用变差。关闭此指标回落规则，不改变22/20/2、等号、回升、止盈止损及冷却。\\n\\n"
        "下一86拟在原第32轮八持仓因子中只加入一项市场流动性代理：过去20个完整日的绝对含分红收益除成交额的平均，再取自然对数并按原训练尺度标准化。只用当时已知交易数据，不补来源、不用未来收益作特征；原周期标签、成熟时钟、等周期权重、岭惩罚1和原进入退出保持。它不是申购、净流入或真实因果价格冲击。加入这一个因子后重训原月度9因子版本，并用自身真实持仓逐日退出，不把旧八项系数误作新九项系数。当前只有方向，尚未登记或训练。\\n\\n"
        "有界检索已发现旧结构估值/风险框架引用etf_amihud_20d。research/cf_dr_rc_own_object_measurement_validity_v1.py中的LIQUIDITY_SHOCK取origin至未来target的最大相对冲击，它是未来评价对象，绝不能直接接入入场或退出。86需直接从当前缓存日行情和成交额重建过去20日代理，先确认原单位和非正成交额、零代理的缺失处理，再固定唯一因子。旧结构框架失败保留，不重开财报、EPS、成分或来源补齐。\\n\\n"
        "先少量查看现有特征构造和九因子月度训练接入点，确认不是已运行同一增量，然后必要测试、一次冻结、直接四账户。无预测显著性前置门，不扫窗口、变换、惩罚或系数。继续优先实际计算及简洁中文交付。\\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第85轮：价格恐慌回落后的进入策略", "",
        "主评价基础／压力净夏普0.315／0.208，较早历史负0.251／负0.357，没有达到1.2。增加了交易机会，未形成可靠收益，停止这项固定规则。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""]+table(result["earlier_diagnostics"])
    lines += ["主每档34个完整周期、68笔成交、178持仓收盘，较早23周期、46笔、86收盘。较早基础价格与分红在扣显式费用之前已亏10,533.70元，另付佣金及滑点6,749.80元，净亏17,283.50元。高读数回落并不足以识别持续回升，不能把失败只归因于费用。", "",
        "主基础年化1.85%、回撤12.97%；较早年化负1.78%、回撤14.93%。两段均无整笔未成交和评价期因子缺失。原文的价格型指标与本项目自定交易规则已区分，结果不支持将其当成可直接使用的底部信号。", "",
        "## 完整中文因子和进出场", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", f"六项必要测试通过；{len(main_f)}行独立低点尺度和指标、{len(early_f)}行因果前缀、四个新账户及{len(cycles)}个含分红周期已核对。下一项只是给原退出模型加入过去20日流动性代理的方向，尚未训练或运行。", ""]
    DOCUMENT.write_text("\\n".join(lines)+"\\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT/p.name)
    shutil.copy2(CONFIG, OUT/"冻结设置.json")
    shutil.copy2(NEXT, OUT/"下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 85, "study": result["study_id"], "title": "价格恐慌波动带回落后进入", "status": status,
        "result": str((RESEARCH/"result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND85_COMPLETE_SYNTHETIC_FEAR_TARGET_NOT_MET",
        count_warning="85轮，359不同设置，375已评价来源版本，380登记含5旧未运行，1298主评价记录。",
        next_work={"status": "PAST_AMIHUD_INCREMENTAL_EXIT_FACTOR_NOT_REGISTERED", "focus": "在原八因子退出中加入现有成交额的过去流动性代理", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="84及85四新账户和关键核对、简洁交付全部完成；86流动性九因子仅方向。")
    index["deliveries"].append({"created_at": now(), "type": "SYNTHETIC_FEAR_ROUND85_FAST_CHINESE_RESULTS", "rounds": [85], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT/"交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "价格分红费用归因": economics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''
    dest.write_text(s, encoding="utf-8")
    print("第85轮关键因子、账户核对与简洁交付脚本已建立。", flush=True)


if __name__ == "__main__":
    main()
