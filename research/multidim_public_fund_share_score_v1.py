"""公募股票及混合基金份额变化的固定历史联合评分。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pdfplumber
import requests

import multidim_nonlinear_score_v1 as common
from multidim_money_surprise_score_v1 import BASE, TREE, fit_tree, return_summary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_public_fund_share_score_v1"
STUDY = "510300_MULTIDIM_PUBLIC_FUND_SHARE_SCORE_V1"
NEW = ["开放式股票基金份额月变", "开放式混合基金份额月变"]
FEATURES = [*BASE, *NEW]
MONTHS = pd.period_range("2023-09", "2025-10", freq="M").astype(str).tolist()
INDEX_URLS = ["https://www.amac.org.cn/sjtj/tjbg/gmjj/", "https://www.amac.org.cn/sjtj/tjbg/gmjj/index_1.html"]
DAILY = ROOT / "reports/research/510300_multidim_financing_composition_v1/data_correction/daily_inputs_corrected_20240808.parquet"
DIV = ROOT / "data/reference/510300_dividends.csv"
REPORT_NAME = "公募份额变化与指数联合评分_历史结果.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return common.digest(path)


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(common.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("公募份额本轮方案已固定，不覆盖。")
    correction = read(ROOT / "config/510300_financing_source_correction_20240808_v1.json")
    if sha(DAILY) != correction["corrected_daily_sha256"]:
        raise ValueError("日线状态不是当前纠正副本。")
    save("protocol.json", {
        "study_id": STUDY, "frozen_at": common.now(), "previous_goal_turn_classification": "PROGRESS_MSCI_INCLUSION_JOINT_STATE_DISCOVERY",
        "question": "开放式股票及混合基金份额存量变化，能否在订单、资金、融资、指数量价状态之外，提供公开后的五日收益信息？",
        "new_evidence": "基金业协会官方月度市场数据，覆盖股票及混合基金总体；与旧510300份额、十只ETF份额规则以及融资余额不同。股票基金仍含ETF，混合基金也非纯股票投资，不把两个字段说成独立现金流。",
        "mechanism": "基金投资者端的份额变化可能反映申赎和发行压力，并与市场涨跌及资金条件共同影响资产配置；基金实际卖买还取决于现金缓冲、股票债券仓位、实物申赎及基金类别变化。这里研究存量状态，不声称识别被迫出售。",
        "months": MONTHS, "source": "中国证券投资基金业协会：公募基金市场数据", "index_urls": INDEX_URLS,
        "sample_reason": "2023年9月及以前文件在2023-11-26集中生成；从该批次最新统计月开始。2025年11月官方表注明不再列示封闭/开放分类，固定截止2025年10月，避免跨口径拼接。",
        "clock": "取官方目录标注日期与PDF文件名P020YYMMDD的日期中较晚者，当日23:59:59才视为可用。文件名日期只是保守版本线索，不证明首次发布时间。若多月同一可用日，只保留最新统计月作为事件，其他月份保留来源但不重复计样本。",
        "source_vintage": "本次下载的官网历史副本，不是当年接收回执；同一PDF里的当前月和上月值用于当时口径的变化，不用未来报告回填。",
        "features_control": BASE, "features_joint": FEATURES,
        "new_feature_definition": "分别为当前份额/同份官方月报中的上月份额-1。单位亿份。份额变化包含新发、净申赎、分红再投、转换、拆并份额及口径因素，不称为纯现金净流入。",
        "asset_decomposition": "以资产净值A、份额Q、平均每份资产v=A/Q，精确分解ΔA=平均v×ΔQ+平均Q×Δv。第二项也含基金构成变化，不能称为纯投资收益；第一项不等于实际现金申赎。",
        "state_time": "可用日当日或之前最后一个A股交易日16:00的纠正后八变量，资金和融资使用其原滞后时钟。",
        "label": {"entry": "可用日之后第一个A股开盘", "open_intervals": 5, "quantity": 10000,
                  "commission": .0004, "minimum": 5., "slippage": .001, "tick": .001,
                  "dividend": "登记日在入场至退出前取得的权益；不是完整账户到账现金。"},
        "model": TREE, "training": {"lookback_sessions": 504, "minimum_mature_events": 12, "label_maturity": "退出开盘不晚于本次可用日决策时刻"},
        "comparisons": "相同事件和训练池的八变量树、十变量树、十变量岭回归alpha10、历史均值。原树复杂度和原成本不变，零参数搜索。",
        "score": "当次预测在当次训练池拟合值中的中位百分位，范围0至100，不是概率。展示五个固定20分档，包含空档。",
        "opportunity": "分数>=80且预测净收益>0；前一机会退出前不再入场。全部机会保存，不凑年度次数。",
        "primary_period": ["2024-01-01", "2025-12-31"], "warmup": "来源较短，前12次成熟观察只训练；不复制月值形成每日独立事件。",
        "account_admission": "共同主要期内，联合树MSE不高于对照、联合高分机会净均值为正且至少新增一个与对照不同的高分入场，三项同时满足才另行固定完整账户；否则封存当前表达，不更改变量、叶数、分数线或持有期。",
        "history_is_independent_validation": False, "new_prospective_forecasts": False, "orders_authorized": False, "goal_achieved": False,
        "inputs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in [DAILY, DIV, ROOT / "research/multidim_money_surprise_score_v1.py"]],
    })
    print("已固定26份同口径报告、两项新字段及原浅树比较，未计算新收益。", flush=True)


def catalog():
    read(OUT / "protocol.json")
    if (OUT / "source_plan.json").exists():
        raise RuntimeError("来源计划已保存，不覆盖。")
    rows, receipts = [], []
    for i, url in enumerate(INDEX_URLS):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path = OUT / "raw" / f"index_{i}.html"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(response.content)
        receipts.append({"url": url, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), "retrieved_at": common.now()})
        soup = BeautifulSoup(response.content, "html.parser")
        for a in soup.find_all("a", href=True):
            title = a.get_text(" ", strip=True)
            match = re.search(r"公募基金市场数据[（(](\d{4})年\s*(\d{1,2})月[）)]", title)
            if not match:
                continue
            month = f"{int(match[1]):04d}-{int(match[2]):02d}"
            if month not in MONTHS:
                continue
            context = a.parent.get_text(" ", strip=True)
            date = re.search(r"\d{4}-\d{2}-\d{2}", context)
            target = urljoin(url, a["href"])
            generated = re.search(r"/P020(\d{6})\d*\.pdf", target)
            if date is None or generated is None:
                continue
            generated_day = pd.to_datetime(generated[1], format="%y%m%d").strftime("%Y-%m-%d")
            rows.append({"stat_month": month, "title": title, "archive_date": date.group(), "pdf_filename_date": generated_day,
                         "available_day": max(date.group(), generated_day), "url": target, "catalog_url": url,
                         "available_at": max(date.group(), generated_day) + "T23:59:59+08:00"})
    unique = {r["stat_month"]: r for r in rows}
    if set(unique) != set(MONTHS):
        raise ValueError("固定月份目录不完整：" + str(sorted(set(MONTHS)-set(unique))))
    save("source_plan.json", sorted(unique.values(), key=lambda r: r["stat_month"]))
    save("catalog_receipt.json", receipts)
    print("已定位26份官方PDF，来源目录及较晚可用日期已保存。", flush=True)


def fetch_parse(item):
    path = OUT / "raw" / (item["stat_month"] + ".pdf")
    if path.exists():
        data = path.read_bytes()
        status = "REUSED_LOCAL_BYTES"
    else:
        attempts = []
        for attempt in range(2):
            try:
                response = requests.get(item["url"], timeout=30)
                response.raise_for_status()
                attempts.append({"attempt": attempt+1, "status": response.status_code, "at": common.now()})
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                attempts.append({"attempt": attempt+1, "error": str(error), "at": common.now()})
                save("network_receipts/"+item["stat_month"]+".json", attempts)
                if attempt == 1:
                    raise
        save("network_receipts/"+item["stat_month"]+".json", attempts)
        data = response.content
        if not data.startswith(b"%PDF"):
            raise ValueError("不是PDF：" + item["stat_month"])
        path.write_bytes(data)
        status = "FETCHED"
    with pdfplumber.open(path) as document:
        content = "\n".join(p.extract_text() or "" for p in document.pages)
        pages = len(document.pages)
    (OUT / "raw" / (item["stat_month"] + ".txt")).write_text(content, encoding="utf-8")
    normalized = re.sub(r"[ \t]+", " ", content)
    title = re.search(r"公募基金市场数据[（(](\d{4})\s*年\s*(\d{1,2})\s*月[）)]", normalized)
    if title is None or f"{int(title[1]):04d}-{int(title[2]):02d}" != item["stat_month"]:
        raise ValueError("PDF统计月与目录不一致：" + item["stat_month"])
    period = pd.Period(item["stat_month"], freq="M")
    dates = re.findall(r"(?:20\d{2})[/.]\d{1,2}[/.]\d{1,2}", normalized)
    distinct = list(dict.fromkeys(pd.Timestamp(x.replace("/", "-")).strftime("%Y-%m-%d") for x in dates))
    expected = [period.end_time.strftime("%Y-%m-%d"), (period - 1).end_time.strftime("%Y-%m-%d")]
    if distinct[:2] != expected or "封闭式基金" not in normalized or "开放式基金" not in normalized:
        raise ValueError("当月上月表头或开放式口径不符：" + item["stat_month"] + str(distinct))
    row = {**item, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), "status": status,
           "retrieved_at": common.now(), "pages": pages, "current_stat_date": expected[0], "previous_stat_date": expected[1]}
    for category, prefix, feature in [("股票基金", "stock", NEW[0]), ("混合基金", "mixed", NEW[1])]:
        lines = [s.strip() for s in normalized.splitlines() if category in s]
        if len(lines) != 1:
            raise ValueError("基金行不唯一：" + item["stat_month"] + category)
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", lines[0])
        if len(numbers) != 6:
            raise ValueError("基金行不是六列：" + item["stat_month"] + lines[0])
        count, shares, assets, old_count, old_shares, old_assets = [float(s.replace(",", "")) for s in numbers]
        if min(count, shares, assets, old_count, old_shares, old_assets) <= 0:
            raise ValueError("基金数据非正。")
        value, old_value = assets/shares, old_assets/old_shares
        share_component = .5*(value+old_value)*(shares-old_shares)
        unit_component = .5*(shares+old_shares)*(value-old_value)
        if abs(share_component+unit_component-(assets-old_assets)) > 1e-7:
            raise ValueError("资产存量分解不一致。")
        row.update({prefix+"_count": int(count), prefix+"_shares_yi": shares, prefix+"_assets_yi": assets,
                    prefix+"_previous_count": int(old_count), prefix+"_previous_shares_yi": old_shares, prefix+"_previous_assets_yi": old_assets,
                    prefix+"_asset_change_yi": assets-old_assets, prefix+"_share_component_yi": share_component,
                    prefix+"_unit_asset_component_yi": unit_component, prefix+"_unit_asset": value, prefix+"_previous_unit_asset": old_value,
                    prefix+"_source_line": lines[0], feature: shares/old_shares-1})
    return row


def sources():
    if (OUT / "source_result.json").exists():
        raise RuntimeError("来源字段已提取，不覆盖。")
    plan = read(OUT / "source_plan.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fetch_parse, plan))
    frame = pd.DataFrame(rows).sort_values("stat_month")
    frame.to_csv(OUT / "全部26个月的官方份额与存量分解.csv", index=False, encoding="utf-8-sig")
    save("parsed_sources.json", rows)
    save("source_result.json", {"completed_at": common.now(), "source_months": len(rows), "original_pdf_files": len(rows),
                                "first_month": MONTHS[0], "last_month": MONTHS[-1],
                                "archive_and_file_date_different": int(frame.archive_date.ne(frame.pdf_filename_date).sum()),
                                "available_days": frame.available_day.nunique(), "source_first_vintage_authenticated": False,
                                "source_sha256": sha(OUT / "parsed_sources.json"), "new_returns_computed": 0})
    print("26份官方月报和两类份额变化已提取，资产规模分解闭合。", flush=True)


def build_events():
    sources = read(OUT / "parsed_sources.json")
    latest = {}
    for row in sources:
        day = row["available_day"]
        if day not in latest or row["stat_month"] > latest[day]["stat_month"]:
            latest[day] = row
    daily = pd.read_parquet(DAILY).sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date).dt.tz_localize(None)
    div = pd.read_csv(DIV)
    div["record_date"] = pd.to_datetime(div.record_date)
    dates = pd.DatetimeIndex(daily.date)
    rows, excluded = [], []
    for source in sources:
        decision = pd.Timestamp(source["available_at"])
        if latest[source["available_day"]]["stat_month"] != source["stat_month"]:
            excluded.append({"stat_month": source["stat_month"], "reason": "同一可用日仅保留最新统计月"})
            continue
        if decision.year > 2025:
            excluded.append({"stat_month": source["stat_month"], "reason": "可用时间晚于固定2025年截止"})
            continue
        state_idx = int(dates.searchsorted(pd.Timestamp(source["available_day"]), side="right"))-1
        state = daily.iloc[state_idx]
        entry_idx, exit_idx = state_idx+1, state_idx+6
        if not np.isfinite(state[BASE].to_numpy(float)).all():
            excluded.append({"stat_month": source["stat_month"], "reason": "八项状态缺失"})
            continue
        for field in ["orders_available_at", "funding_available_at"]:
            if pd.Timestamp(state[field]) > decision:
                raise ValueError("状态信息晚于决策。")
        a,b = daily.iloc[entry_idx],daily.iloc[exit_idx]
        if a.date <= pd.Timestamp(source["available_day"]):
            raise ValueError("事件日使用了同日开盘。")
        buy = np.ceil(float(a.open)*1.001*1000-1e-9)/1000
        sell = np.floor(float(b.open)*.999*1000+1e-9)/1000
        entitlement = float(div.loc[div.record_date.ge(a.date)&div.record_date.lt(b.date),"cash_dividend_per_share"].sum())
        paid = 10000*buy+max(5.,10000*buy*.0004)
        received = 10000*(sell+entitlement)-max(5.,10000*sell*.0004)
        rows.append({**source, **{f:float(state[f]) for f in BASE}, "decision_at": decision, "state_date": state.date,
                     "state_idx": state_idx, "entry_idx": entry_idx, "exit_idx": exit_idx,
                     "entry_date": a.date.strftime("%Y-%m-%d"), "exit_date": b.date.strftime("%Y-%m-%d"),
                     "exit_at": b.date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30),
                     "entry_open": a.open, "exit_open": b.open, "dividend_per_share": entitlement,
                     "buy_with_slippage":buy,"sell_with_slippage":sell,"cash_in":paid,"cash_out":received,
                     "actual_net5":received/paid-1,"gross_return":(float(b.open)+entitlement)/float(a.open)-1,
                     "era":"2024—2025" if decision.year>=2024 else "预热2023"})
    events = pd.DataFrame(rows).sort_values("decision_at").reset_index(drop=True)
    events.to_parquet(OUT / "月度事件联合状态及标签.parquet",index=False)
    events.to_csv(OUT / "全部事件状态及标签.csv",index=False,encoding="utf-8-sig")
    save("未准入事件.json",excluded)
    return events


def run():
    if (OUT / "result.json").exists() or (OUT / "逐事件联合评分.csv").exists():
        raise RuntimeError("本轮已有评分结果，不重跑。")
    protocol=read(OUT / "protocol.json")
    source=read(OUT / "source_result.json")
    for item in protocol["inputs"]:
        if sha(ROOT/item["path"])!=item["sha256"]:
            raise ValueError("固定输入变化。")
    if sha(OUT/"parsed_sources.json")!=source["source_sha256"]:
        raise ValueError("来源字段变化。")
    for item in read(OUT/"parsed_sources.json"):
        if sha(ROOT/item["path"])!=item["sha256"]:
            raise ValueError("官方PDF副本变化。")
    events=build_events()
    rows,models,warmup=[],[],[]
    for _,current in events.iterrows():
        pool=events[events.state_idx.ge(current.state_idx-504)&events.exit_at.le(current.decision_at)]
        if len(pool)<12:
            warmup.append({"stat_month":current.stat_month,"available_at":current.available_at,"mature_events":len(pool)})
            continue
        if pool.stat_month.eq(current.stat_month).any():
            raise ValueError("训练含当前标签。")
        sp,ss,sm=fit_tree(pool,current,BASE)
        jp,js,jm=fit_tree(pool,current,FEATURES)
        x,y=pool[FEATURES].to_numpy(float),pool.actual_net5.to_numpy(float)
        mean,scale=x.mean(axis=0),x.std(axis=0,ddof=1)
        scale[scale<1e-12]=1.
        z=np.clip((x-mean)/scale,-5,5)
        zmean=z.mean(axis=0)
        centered=z-zmean
        coef=np.linalg.solve(centered.T@centered+10*np.eye(len(FEATURES)),centered.T@(y-y.mean()))
        linear=float(y.mean()+(np.clip((current[FEATURES].to_numpy(float)-mean)/scale,-5,5)-zmean)@coef)
        rows.append({**current.to_dict(),"training_events":len(pool),"training_months":";".join(pool.stat_month),"training_max_exit_at":pool.exit_at.max(),
                     "state_prediction":sp,"state_score":ss,"joint_prediction":jp,"joint_score":js,
                     "linear_prediction":linear,"mean_prediction":float(y.mean()),"prediction_changed":abs(sp-jp)>1e-12,
                     "new_features_used":"；".join(f for f in NEW if f in jm["used_features"])})
        models.append({"stat_month":current.stat_month,"decision_at":current.decision_at,"training_months":pool.stat_month.to_list(),
                       "state_tree":sm,"joint_tree":jm,"linear":{"mean":mean.tolist(),"scale":scale.tolist(),"zmean":zmean.tolist(),"coef":coef.tolist(),"ymean":float(y.mean())}})
    predictions=pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("固定来源没有12次成熟训练观察，评分不运行。")
    predictions.to_csv(OUT/"逐事件联合评分.csv",index=False,encoding="utf-8-sig")
    save("saved_models.json",models)
    save("训练预热.json",warmup)
    errors=[{"model":m,"n":len(predictions),"mse":float(((predictions[m+"_prediction"]-predictions.actual_net5)**2).mean())} for m in ["state","joint","linear","mean"]]
    buckets,opportunities=[],[]
    for m in ["state","joint"]:
        band=np.minimum((predictions[m+"_score"]/20).astype(int),4)
        for i in range(5):
            buckets.append({"model":m,"score_band":f"{20*i}—{20*(i+1)}",**return_summary(predictions.loc[band.eq(i),"actual_net5"])})
        next_decision=pd.Timestamp("1900-01-01",tz="Asia/Shanghai")
        for _,row in predictions.iterrows():
            if row.decision_at<next_decision or row[m+"_score"]<80 or row[m+"_prediction"]<=0:
                continue
            opportunities.append({"model":m,"stat_month":row.stat_month,"era":row.era,"signal_date":row.decision_at,
                                  "entry_date":row.entry_date,"exit_date":row.exit_date,"score":row[m+"_score"],
                                  "prediction":row[m+"_prediction"],"net_return":row.actual_net5})
            next_decision=row.exit_at
    opp=pd.DataFrame(opportunities,columns=["model","stat_month","era","signal_date","entry_date","exit_date","score","prediction","net_return"])
    summaries=[{"model":m,**return_summary(opp.loc[opp.model.eq(m),"net_return"])} for m in ["state","joint"]]
    annual=[]
    for m in ["state","joint"]:
        for year in [2024,2025]:
            selected=opp[opp.model.eq(m)&pd.to_datetime(opp.entry_date).dt.year.eq(year)]
            annual.append({"model":m,"year":year,**return_summary(selected.net_return)})
    extra=opp[opp.model.eq("joint")&~opp.entry_date.isin(opp.loc[opp.model.eq("state"),"entry_date"])]
    gate={"joint_mse_not_worse":errors[1]["mse"]<=errors[0]["mse"],"joint_high_score_net_mean_positive":bool(summaries[1]["n"]>0 and summaries[1]["mean_net5"]>0),"new_high_score_entry":len(extra)>0}
    admitted=all(gate.values())
    for name,frame in [("固定五档比较.csv",pd.DataFrame(buckets)),("四项模型误差.csv",pd.DataFrame(errors)),
                       ("全部固定高分机会.csv",opp),("高分机会比较.csv",pd.DataFrame(summaries)),("高分机会逐年次数.csv",pd.DataFrame(annual))]:
        frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    result={"study_id":STUDY,"completed_at":common.now(),"status":"COMPLETED_PUBLIC_FUND_SHARE_JOINT_SCORE",
            "candidate_status":"ACCOUNT_COMPARISON_REQUIRED" if admitted else "REJECTED_FIXED_INCREMENT_NO_PARAMETER_RESCUE",
            "source_months":len(MONTHS),"admitted_events":len(events),"warmup_events":len(warmup),"scored_events":len(predictions),
            "first_scored_entry":predictions.entry_date.min(),"last_scored_entry":predictions.entry_date.max(),
            "new_model_families":3,"model_fits":3*len(predictions),"parameter_searches":0,
            "changed_predictions":int(predictions.prediction_changed.sum()),"new_fields_used_events":int(predictions.new_features_used.ne("").sum()),
            "model_errors":errors,"mse_relative_change":errors[1]["mse"]/errors[0]["mse"]-1,
            "buckets":buckets,"opportunity_summary":summaries,"annual_opportunities":annual,
            "new_high_entries":extra.to_dict("records"),"account_admission_conditions":gate,"account_admitted":admitted,
            "new_accounts":0,"account_sharpe":"NOT_COMPUTED","goal_achieved":False,"orders_authorized":False,"current_view":"NO_VIEW"}
    save("result.json",result)
    print(json.dumps(common.clean({k:result[k] for k in ["scored_events","changed_predictions","mse_relative_change","opportunity_summary","account_admission_conditions","candidate_status"]}),ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","catalog","sources","run"])
    globals()[parser.parse_args().stage]()
