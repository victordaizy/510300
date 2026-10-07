"""复用当月统计局原文，对订单与库存扩散差做固定月度账户初筛。"""
import argparse
import importlib.util
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from research import factor96_rapid_feasibility_v1 as rapid
from scripts.record_factor96_remaining_changes_v1 import update

ROOT, core = rapid.ROOT, rapid.core
OUT = ROOT / "reports/research/510300_factor96_rapid_orders_inventory_v1"
STUDY = "510300_FACTOR96_RAPID_ORDERS_INVENTORY_V1"
SOURCE = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet"
START, END, HOLD = "2017-01-03", "2025-12-31", 20
FACTORS = ["ORDERS_INVENTORY_GAP", "GAP_CHANGE3"]
POLICIES = [f"{key}_{side}" for key in FACTORS for side in ["HIGH", "LOW"]] + ["RELEASE_ALL", "PRICE_CONTROL", "BUY_HOLD_50", "CASH"]
read, save = rapid.read, rapid.save


def source_index():
    d = pd.read_parquet(SOURCE)
    d = d.loc[d.reference_period.between("2015-01", "2025-12")].sort_values("reference_period").reset_index(drop=True)
    assert d.reference_period.tolist() == [str(v) for v in pd.period_range("2015-01", "2025-12", freq="M")]
    return d


def freeze():
    assert not OUT.exists(), "本研究已存在，禁止覆盖结果"
    sources = source_index()
    for row in sources.itertuples():
        assert core.digest(ROOT / row.raw_path) == row.source_hash
    save(OUT / "protocol.json", {
        "at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "previous_goal_turn": "PROGRESS_32_STRUCTURE_ACCOUNTS_COMPLETED_NO_PROMISING_CANDIDATE",
        "user_priority": "只交易510300及现金；先简单检验可行性，有效果再展开；不需要交付包。",
        "economic_question": "制造业需求相对成品库存的扩散差，以及其三个月变化，是否提供未来约一月的ETF方向信息。扩散指数差不是订单实际数量减库存，也不等于市场预期差。",
        "source_scope": "132份已存2015至2025年当月PMI原文；仅取当月行，不使用后续月报中的旧月份行补值。与已存新订单字段逐月交叉核对；不新增网络采集。",
        "parser": "制造业新订单由含生产/原材料库存/从业人员的表头及当月行匹配；产成品库存由同名唯一表头匹配；保留原表行和重复桌面/移动表格的一致性。",
        "vintage_limit": "本地保存的是后来取得的当期报告页面，不能证明不可变的历史首次版本；不使用latest_revised_value字段。",
        "formulas": {"ORDERS_INVENTORY_GAP": "制造业新订单扩散指数减产成品库存扩散指数，百分点，保留一位小数。",
                     "GAP_CHANGE3": "当月扩散差减三个月前扩散差；三个间隔内分类标准及样本口径须相同。"},
        "thresholds": "每个变量仅HIGH>=此前两日历年70%分位、LOW<=30%分位；仅用同分类版本/行业数/样本企业数的已公布月份，至少12个先前月报。当前月不进入阈值。两变量均满足资料门才进入全部候选和公布日对照。",
        "clock": "沿用已核实原文公布时钟，统一公布日末形成判断；非交易日公布的资料保守映射到随后首个交易日收盘，再下一开盘交易。严禁按统计月提前使用。",
        "decision_frequency": "每份新月报最多一次入场请求；持仓内忽略新信号，不把月度观测填成多次独立新闻。",
        "execution": "次开盘入场，持有20个开盘间隔；复用既有50%最大仓位、ES、回撤储备、10%收盘回撤后永久退出、100份整手、T+1、涨跌停、分红现金与应收及成本账本。",
        "controls": {"RELEASE_ALL": "同样可用的每份月报均产生入场请求。", "PRICE_CONTROL": "仅当同月报可用日ETF过去5日总收益为正时入场。", "BUY_HOLD_50": "初始半仓买入持有背景，不使用候选的风险停止。", "CASH": "零利息现金。"},
        "costs": core.COSTS, "capital": [200000, 20000], "annual_days": 242,
        "sample": [START, END], "periods": [[START, "2020-12-31"], ["2021-01-01", END]],
        "planned_accounts": 32, "new_directions": 4,
        "correlation_diagnostic": "两个变量各与下一合法开盘至20开盘间隔后的含分红固定份额收益计算Spearman，全期及两个固定子期均列出。少量相邻月标签可能重叠；不报独立日显著性，不称独立验证。",
        "followup_filter": "20万STRESS夏普>=0.8、CAGR>=5%、回撤<=10%、入场>=20、两子期收益正，且夏普/CAGR均高于RELEASE_ALL。只用于决定是否深化。",
        "final_target": "完整账户成本后夏普>=1.2，并联合检验CAGR>=10%、回撤<=10%；独立验证未完成不得正式达标。",
        "legacy_boundary": "新变量包含此前未进入账户的产成品库存差；不是重新优化旧新订单水平、货币数量、宏观意外或购进出厂价格传导规则，全部旧失败保留。",
        "no_rescue": "不在看本轮绩效后更换方向、阈值、持有期、成本、年份或训练窗口。",
        "official_reference": "https://www.stats.gov.cn/sj/zxfb/202501/t20250127_1958493.html",
        "orders_authorized": False, "delivery_package_required": False,
    })
    sources.to_parquet(OUT / "source_index.parquet", index=False)
    paths = [Path(__file__), Path(rapid.__file__), Path(core.__file__), rapid.ENGINE_PATH, SOURCE,
             rapid.PRICE_BASE / "market.parquet", rapid.PRICE_BASE / "price_features.parquet", rapid.PRICE_BASE / "dividends.csv",
             OUT / "protocol.json", OUT / "source_index.parquet"]
    files = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    files += [{"path": str(ROOT / r.raw_path), "sha256": r.source_hash} for r in sources.itertuples()]
    save(OUT / "freeze.json", {"at": core.now(), "files": files})
    print("已冻结订单库存差的4个方向和4个对照，共32账户；仅复用132份本地月报。", flush=True)


def compact(text):
    return re.sub(r"\s+", "", text)


def parse_month(row):
    raw = (ROOT / row.raw_path).read_bytes()
    soup = BeautifulSoup(raw, "html.parser")
    text = compact(soup.get_text())
    year, month = map(int, row.reference_period.split("-"))
    target = f"{year}年{month}月"
    titles = [compact(t.get_text()) for t in soup.find_all(["title", "h1", "h2"])]
    assert any(target in t and "采购经理指数" in t for t in titles), (row.reference_period, "标题月份不符")
    extracted = {"new_orders": [], "finished_inventory": []}
    anchors = []
    for table_number, table in enumerate(soup.find_all("table")):
        rows = [[compact(cell.get_text()) for cell in tr.find_all(["td", "th"], recursive=False)] for tr in table.find_all("tr")]
        current = [r for r in rows if r and r[0] == target]
        for key, term in [("new_orders", "新订单"), ("finished_inventory", "产成品库存")]:
            headers = [r for r in rows if term in r and (key != "new_orders" or all(v in r for v in ["生产", "原材料库存", "从业人员"]))]
            if not headers:
                continue
            assert len(headers) == 1 and len(current) == 1, (row.reference_period, key, "表头或当月行不唯一")
            header, values = headers[0], current[0]
            offset = len(values)-len(header)
            assert offset in [0, 1, 2]
            column = header.index(term)+offset
            cell = values[column]
            assert re.fullmatch(r"\d{1,2}(?:\.\d)?", cell), (row.reference_period, key, cell)
            value = float(cell)
            assert 0 < value < 100
            extracted[key].append(value)
            anchors.append({"key": key, "table_index": table_number, "header": header, "current_month_row": values, "value_column": column})
    assert all(v and len(set(v)) == 1 for v in extracted.values()), (row.reference_period, extracted)
    orders, inventory = extracted["new_orders"][0], extracted["finished_inventory"][0]
    assert abs(orders-float(row.first_release_value)) < 1e-10, (row.reference_period, "新订单与继承字段不符")
    samples = set(re.findall(r"(?<!非)制造业的(\d+)个行业大类[，,](\d+)家调查样本", text))
    standards = set(re.findall(r"GB/T4754[—－-](20\d{2})", text))
    assert len(samples) == len(standards) == 1, (row.reference_period, "口径信息不唯一")
    industries, firms = next(iter(samples))
    standard = next(iter(standards))
    assert "经季节调整" in text
    published = pd.Timestamp(row.published_at)
    assert published.tzinfo is not None
    assert str(published.tz_localize(None).to_period("M")) in [row.reference_period, str(pd.Period(row.reference_period, freq="M")+1)]
    return {"stat_month": row.reference_period, "published_at": published,
            "known_at": published.normalize()+pd.Timedelta(days=1)-pd.Timedelta(seconds=1),
            "new_orders": orders, "finished_inventory": inventory, "ORDERS_INVENTORY_GAP": round(orders-inventory, 1),
            "method_group": f"GB{standard}_{industries}IND_{firms}FIRMS", "source_url": row.source_url,
            "raw_path": row.raw_path, "raw_sha256": row.source_hash, "anchors": anchors,
            "historical_first_vintage_verified": False, "current_month_row_only": True}


def monthly_features(raw):
    q = raw.sort_values("known_at").reset_index(drop=True).copy()
    periods = pd.PeriodIndex(q.stat_month, freq="M")
    assert periods.is_monotonic_increasing and periods.is_unique
    same_method = q.method_group.eq(q.method_group.shift(3)) & q.method_group.eq(q.method_group.shift(1)) & q.method_group.eq(q.method_group.shift(2))
    consecutive = pd.Series(periods.asi8).diff(3).eq(3)
    q["GAP_CHANGE3"] = q.ORDERS_INVENTORY_GAP.diff(3).round(1).where(same_method & consecutive)
    for key in FACTORS:
        q[key+"_q30"], q[key+"_q70"] = np.nan, np.nan
        q[key+"_history_count"] = 0
        for i in range(len(q)):
            history = q.iloc[:i]
            known = history.known_at.ge(q.known_at.iloc[i]-pd.DateOffset(years=2)) & history.method_group.eq(q.method_group.iloc[i])
            values = history.loc[known, key].dropna()
            q.loc[i, key+"_history_count"] = len(values)
            if len(values) >= 12:
                q.loc[i, [key+"_q30", key+"_q70"]] = values.quantile([.3, .7]).to_numpy()
        q[key+"_valid"] = q[key].notna() & q[key+"_q70"].gt(q[key+"_q30"])
    q["eligible"] = q[[key+"_valid" for key in FACTORS]].all(axis=1)
    for key in FACTORS:
        q[key+"_HIGH"] = q.eligible & q[key].ge(q[key+"_q70"])
        q[key+"_LOW"] = q.eligible & q[key].le(q[key+"_q30"])
    return q


def daily_signals(m, risk, monthly):
    q = monthly.copy()
    calendar = pd.DatetimeIndex(m.date)
    publication_dates = pd.DatetimeIndex(q.known_at).tz_localize(None).normalize()
    pos = calendar.searchsorted(publication_dates)
    keep = pos < len(calendar)
    q = q.loc[keep].copy()
    q["date"] = calendar[pos[keep]]
    assert q.date.is_unique
    assert (q.known_at <= q.date.dt.tz_localize("Asia/Shanghai")+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)).all()
    columns = ["date", "stat_month", "known_at", "eligible", *FACTORS, *POLICIES[:4]]
    f = pd.DataFrame({"date": m.date, "es95": risk.es95}).merge(q[columns], on="date", how="left", validate="one_to_one")
    for key in POLICIES[:4]:
        f[key] = f[key].eq(True)
    f["RELEASE_ALL"] = f.eligible.eq(True)
    r = np.log((m.close+m.dividend)/m.close.shift())
    f["price_return5"] = r.rolling(5).sum().to_numpy()
    f["PRICE_CONTROL"] = f.RELEASE_ALL & f.price_return5.gt(0)
    return f


def correlation(f, m, dividends):
    records = []
    for i in np.flatnonzero(f.RELEASE_ALL & f.date.between(START, END)):
        entry, exit_idx = i+1, i+1+HOLD
        if exit_idx >= len(m):
            continue
        a, b = m.date.iloc[entry], m.date.iloc[exit_idx]
        entitled = dividends.record_date.ge(a) & dividends.record_date.lt(b) & dividends.ex_date.le(b)
        gross = (m.open.iloc[exit_idx]-m.open.iloc[entry]+dividends.loc[entitled, "cash_dividend_per_share"].sum())/m.open.iloc[entry]
        records.append({"date": f.date.iloc[i], "entry_date": a, "exit_date": b, "forward_gross_return20": float(gross),
                        **{key: float(f[key].iloc[i]) for key in FACTORS}})
    labels = pd.DataFrame(records)
    labels.to_parquet(OUT / "monthly_diagnostic_labels.parquet", index=False)
    rows = []
    for key in FACTORS:
        for label, start, end in [("full", START, END), ("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
            part = labels.loc[labels.date.between(start, end)]
            rows.append({"factor": key, "period": label, "observations": len(part),
                         "spearman": float(part[key].corr(part.forward_gross_return20, method="spearman")) if len(part) > 2 else None})
    save(OUT / "correlation_diagnostic.json", {"at": core.now(), "rows": rows, "independent_validation": False,
                                               "overlapping_labels_possible": True, "direction_selection_after_result": False})
    return rows


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(row["path"])) == row["sha256"]
    save(OUT / "run_started.json", {"at": core.now(), "planned_accounts": 32})
    records = [parse_month(row) for row in source_index().itertuples()]
    save(OUT / "source_fields_and_anchors.json", records)
    q = pd.DataFrame([{k: v for k, v in row.items() if k != "anchors"} for row in records])
    q.to_parquet(OUT / "released_orders_inventory.parquet", index=False)
    monthly = monthly_features(q)
    monthly.to_parquet(OUT / "monthly_features.parquet", index=False)
    m = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    m.date = pd.to_datetime(m.date)
    m = m.loc[m.date.le(END)].reset_index(drop=True)
    risk = pd.read_parquet(rapid.PRICE_BASE / "price_features.parquet")
    risk.date = pd.to_datetime(risk.date)
    assert m.date.equals(risk.date) and len(m) == 3307
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    f = daily_signals(m, risk, monthly)
    f.to_parquet(OUT / "daily_signals.parquet", index=False)
    cutoff = pd.Timestamp("2021-12-31")
    earlier = q.loc[q.known_at.lt((cutoff+pd.Timedelta(days=1)).tz_localize("Asia/Shanghai"))].copy()
    short_monthly = monthly_features(earlier)
    pd.testing.assert_frame_equal(monthly.loc[monthly.known_at.le(earlier.known_at.max())].reset_index(drop=True), short_monthly)
    short_f = daily_signals(m.loc[m.date.le(cutoff)].copy(), risk.loc[risk.date.le(cutoff)].copy(), short_monthly)
    pd.testing.assert_frame_equal(f.loc[f.date.le(cutoff)].reset_index(drop=True), short_f)
    spec = importlib.util.spec_from_file_location("orders_inventory_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    rapid.START, rapid.END, rapid.HOLD = START, END, HOLD
    accounts, annual = [], []
    for capital in [200000, 20000]:
        for cost in ["BASE", "STRESS"]:
            for policy in POLICIES:
                ledger, orders = rapid.simulate(m, f, dividends, policy, capital, cost, engine)
                ledger["reason"] = ledger.reason.replace({"十个开盘间隔到期": "二十个开盘间隔到期"})
                if not orders.empty:
                    orders["reason"] = orders.reason.replace({"十个开盘间隔到期": "二十个开盘间隔到期"})
                folder = OUT / "accounts" / f"{capital}_{cost}_{policy}"
                folder.mkdir(parents=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                orders.to_parquet(folder / "orders.parquet", index=False)
                row = core.metrics(ledger, capital)
                row.update(policy=policy, capital=capital, cost=cost,
                           entries=int(((ledger.filled_quantity > 0) & ledger.shares_before.eq(0)).sum()),
                           stopped_at=ledger.loc[ledger.risk_stopped, "date"].min().isoformat() if ledger.risk_stopped.any() else None,
                           ledger_path=str((folder / "ledger.parquet").relative_to(ROOT)))
                for label, a, b in [("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
                    row.update({label+"_"+k: v for k, v in rapid.period_metrics(ledger, a, b).items()})
                accounts.append(row)
                for year in range(2017, 2026):
                    annual.append({"policy": policy, "capital": capital, "cost": cost, "year": year,
                                   **rapid.period_metrics(ledger, f"{year}-01-01", f"{year}-12-31")})
            print(f"已完成{capital}元、{cost}成本的8个订单库存账户。", flush=True)
    rapid.HOLD = 10
    table = pd.DataFrame(accounts)
    table["new_candidate"] = table.policy.isin(POLICIES[:4])
    table["beats_release_control"] = False
    for (capital, cost), part in table.groupby(["capital", "cost"]):
        baseline = part.loc[part.policy.eq("RELEASE_ALL")].iloc[0]
        table.loc[part.index, "beats_release_control"] = part.net_sharpe.gt(baseline.net_sharpe) & part.cagr.gt(baseline.cagr)
    table["joint_historical_target"] = table.new_candidate & table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1)
    table["worth_followup"] = (table.new_candidate & table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1)
                               & table.entries.ge(20) & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0)
                               & table.beats_release_control)
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    primary.to_csv(OUT / "20万元压力成本比较.csv", index=False, encoding="utf-8-sig")
    diagnostic = correlation(f, m, dividends)
    save(OUT / "result.json", {"at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "account_scenarios": len(table), "new_directions": 4, "source_reports": len(q),
        "source_method_counts": q.method_group.value_counts().to_dict(), "eligible_release_decisions": int(f.RELEASE_ALL[f.date.between(START, END)].sum()),
        "primary_results": primary.to_dict("records"), "correlation_diagnostic": diagnostic,
        "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(),
        "joint_historical_target_count": int(primary.joint_historical_target.sum()), "qualified_candidates": 0,
        "causal_prefix_check": "PASS", "historical_first_vintage_verified": False, "independent_forward_observations": 0,
        "goal_achieved": False, "new_source_downloads": 0, "orders_authorized": False, "delivery_package_created": False})
    print(primary[["policy", "net_sharpe", "cagr", "max_drawdown", "entries", "worth_followup"]].to_string(index=False), flush=True)
    print("月度相关性诊断：", diagnostic, flush=True)


def record():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    table = pd.read_csv(OUT / "all_account_metrics.csv")
    assert len(table) == 32
    for row in table.itertuples():
        ledger = pd.read_parquet(ROOT / row.ledger_path)
        actual = core.metrics(ledger, row.capital)
        for key in ["cagr", "max_drawdown", "end_equity", "commission", "slippage"]:
            assert np.isclose(actual[key], getattr(row, key), rtol=1e-10, atol=1e-8)
        if pd.notna(row.net_sharpe):
            assert np.isclose(actual["net_sharpe"], row.net_sharpe, atol=1e-10)
    save(OUT / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_32_SAVED_ACCOUNT_RECOMPUTATIONS_AND_CAUSAL_PREFIX",
                                                   "source_new_orders_crosschecks": 132, "max_accounting_error": float(table.max_identity_error.max())})
    names = {"ORDERS_INVENTORY_GAP_HIGH": "订单库存差高", "ORDERS_INVENTORY_GAP_LOW": "订单库存差低",
             "GAP_CHANGE3_HIGH": "三个月变化高", "GAP_CHANGE3_LOW": "三个月变化低", "RELEASE_ALL": "每份月报入场对照",
             "PRICE_CONTROL": "月报日价格对照", "BUY_HOLD_50": "半仓买入持有背景", "CASH": "现金"}
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    promising = result["worth_followup"]
    decision = "没有新方向通过初筛，本轮四条简单用法结束，不据结果调参。" if not promising else "待深入候选："+"、".join(promising)+"，尚未达到正式验收。"
    lines = ["只交易510300及现金、成本后夏普1.2目标不变。本轮复用132份当月统计局原文，检验订单库存扩散差及三个月变化，共4个方向、32个账户。",
             decision, "2017至2025年，20万元压力成本：", "| 方向 | 净夏普 | 年化收益 | 最大回撤 | 入场次数 |", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        value = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "无交易"
        lines.append(f"| {names[row.policy]} | {value} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    lines += ["", "两个变量对公布后下一合法开盘起20个开盘间隔收益的Spearman：", "| 变量 | 时段 | 月度观测 | 秩相关 |", "|---|---|---:|---:|"]
    for row in result["correlation_diagnostic"]:
        lines.append(f"| {row['factor']} | {row['period']} | {row['observations']} | {row['spearman']:.3f} |")
    lines += ["", "每份月报只取本月行，跨月发布按公布日使用；132项新订单与继承字段逐项相符。阈值只取此前两日历年同口径月报，三个月变化不跨口径变更。按宏观频率固定持有20个开盘间隔，含全部现金日、分红、成本、整手和T+1。",
              "这些为后来保存的当期网页，不能证明历史首次版本不可变；相关系数样本少、相邻标签可能重叠，全部属于历史探索。未重新运行旧新订单水平、购进出厂价格传导或其他失败宏观规则。",
              "统计含义依据[统计局PMI月报](https://www.stats.gov.cn/sj/zxfb/202501/t20250127_1958493.html)：这里使用两个扩散指数的百分点差，不是订单和库存实际数量之差。",
              "账本复算32账户，截断未来信息的特征和信号检查一致。没有交付包、没有恢复采集任务、没有交易执行。"]
    (OUT / "快速检验结论.md").write_text("\n\n".join(lines[:3])+"\n\n"+"\n".join(lines[3:]), encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "factor_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]
    status, factors, mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RAPID_STRUCTURE_V1"
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix()
    note = f"订单库存扩散差及三个月变化共4方向、32账户完成，待深入候选{len(promising)}，正式达标0。132份已有当月原文可复用，无新增资料采集。"
    status.update(at=core.now(), latest_round=STUDY, latest_result=relative+"/result.json", last_research_result=note,
        last_completed_account_experiment=STUDY, last_completed_account_result=relative+"/result.json",
        latest_progress_receipt=relative+"/saved_verification_receipt.json", latest_continuation_classification="PROGRESS_32_ORDERS_INVENTORY_ACCOUNTS",
        admitted_account_scenarios_this_round=32, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=status["cumulative_admitted_account_scenarios"]+32,
        cumulative_executed_account_scenarios=status["cumulative_executed_account_scenarios"]+32,
        rapid_screen_direction_tests=status["rapid_screen_direction_tests"]+4,
        current_research_phase="DEEPEN_PROMISING_MONTHLY_INFORMATION" if promising else "MONTHLY_ORDERS_INVENTORY_SIMPLE_RULES_ENDED",
        next_independent_source_action="DEEPEN_FROZEN_PROMISING_MONTHLY_SIGNAL" if promising else "REASSESS_MECHANISM_BASED_ON_ACCUMULATED_NEGATIVE_ACCOUNT_RESULTS",
        goal_status="active", goal_achieved=False, qualified_candidates=[], source_detail_work_deferred_by_user=True)
    if not promising:
        status["stopped_simple_research_branches"].append("4_MONTHLY_ORDERS_INVENTORY_DIRECTIONS")
    for row in factors:
        if row["id"] == "K04":
            row["rapid_monthly_screen"] = {"study_id": STUDY, "result": relative+"/result.json", "status": "PROMISING_REQUIRES_REVIEW" if promising else "NO_PROMISING_FIXED_SIMPLE_RULE", "historical_first_vintage_verified": False}
    mandate.update(current_round=STUDY, current_protocol=relative+"/protocol.json", last_research_result=note,
        latest_progress_receipt=relative+"/saved_verification_receipt.json", latest_continuation_report=relative+"/快速检验结论.md",
        latest_continuation_classification=status["latest_continuation_classification"], research_execution_state=status["current_research_phase"],
        goal_status="active", goal_achieved=False)
    for path, obj in zip(paths, objects):
        update(path, core.clean(obj))
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": core.now(), "before": before,
        "after": [{"path": str(p), "sha256": core.digest(p)} for p in paths], "new_accounts": 32, "goal_achieved": False})
    print(note, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run", "record"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "run": run, "record": record}[stage]()
