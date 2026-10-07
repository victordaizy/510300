"""按周报自然刊期补齐尚缺的月份，不改变已准入来源，不读取收益。"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path

import pandas as pd

import collect_bualuang_money_consensus_v1 as base
import collect_nbg_money_consensus_v1 as nbg

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_money_consensus_source_extension_v1"
OUT = PARENT / "nbg_weekly"
NEW_HOST = "https://www.nbg.gr/-/jssmedia/Files/Group/meletes-oikonomikes-analuseis/diethneis-agores-oikonomia/ebdomadiaia-episkopisi-diethnwn-agorwn/"


def freeze() -> None:
    if OUT.exists():
        raise RuntimeError("周报自然刊期补齐协议已存在，不覆盖。")
    for folder in ("inputs", "raw", "receipts", "parsed", "results", "figures", "code"):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    existing = {}
    for source, directory in [("Bualuang",PARENT),("OCBC",PARENT/"ocbc"),("NBG_FIVE_DAY",PARENT/"nbg")]:
        for row in json.loads((directory/"results/admitted_pairs.json").read_text(encoding="utf-8")):
            existing.setdefault(row["stat_month"],source)
    official = pd.read_csv(PARENT/"inputs/official_releases.csv")
    chosen = official[~official.stat_month.isin(existing)]
    rows = []
    for r in chosen.itertuples():
        release = pd.Timestamp(r.published_at).tz_localize(None).normalize()
        day = release-pd.Timedelta(days=1)
        while day.weekday()!=1:
            day-=pd.Timedelta(days=1)
        if day.year < 2022:
            filenames=[("short_year","gmr_"+day.strftime("%d-%m-%y")+".pdf",nbg.HOST),
                       ("long_year","gmr_"+day.strftime("%d-%m-%Y")+".pdf",nbg.HOST)]
        else:
            filenames=[("new_archive","NBG-GlobalMarketsRoundup-"+day.strftime("%d-%m-%Y")+".pdf",NEW_HOST)]
        for variant, filename, host in filenames:
            rows.append({"stat_month":r.stat_month,"release_at":r.published_at,"offset_days":int((release-day).days),
                "report_date":str(day.date()),"variant":variant,"filename":filename,"url":host+filename})
    pd.DataFrame(rows).to_csv(OUT/"inputs/candidate_urls.csv",index=False,encoding="utf-8-sig")
    base.save(OUT/"protocol.json",{
        "study_id":"NBG_WEEKLY_NATURAL_PUBLICATION_COMPLETION_V1","frozen_at":base.now(),
        "reason":"日报固定五日检索和旧NBG固定五日检索完成后，多个周一/周二央行公布事件没有五日内周二刊期。另以周报实际刊期规则补齐所有未准入月份；旧协议和结果保留。",
        "universe":"104个月中尚缺的全部月份，不按上涨、预期差方向或拟合结果选择；即使达到样本门槛也完成整份清单",
        "known_before_extension":{"old_definition_pairs":sum(m<'2025-01' for m in existing),"new_definition_pairs":sum(m>='2025-01' for m in existing),"existing_pairs":len(existing)},
        "weekly_rule":"只查央行公布日前最近一期周二报告，距公布1至7自然日；不查更早刊期，不收同日或事后报告。与1至5日规则的区别仅为来源刊期，五交易日收益窗口不变。",
        "source_priority":"Bualuang、OCBC、原NBG五日档案、此周报补齐依次填缺；已有数据不替换，不按数值或结果优选。",
        "admission":"沿用封面、国家、所属月、调查/实际/前值及M1口径门；原始报告今日回收仍仅为历史重建。",
        "all_original_gates_unchanged":True,"no_model_or_market_returns_read_for_extension":True,
        "model_minimum":"每口径36训练加至少24评价；不得因距离门槛较近而降低。",
        "forecast_age_limit_days":7,"candidate_urls":len(rows),"missing_months":len(chosen),
        "caution":"来源日期差异和缺失机制均保留；本次扩展也属于上游研究选择历史，不构成独立验证。"})
    base.save(OUT/"freeze_receipt.json",{"protocol_sha256":base.digest(OUT/"protocol.json"),
        "candidates_sha256":base.digest(OUT/"inputs/candidate_urls.csv"),"official_sha256":base.digest(PARENT/"inputs/official_releases.csv")})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print(f"已冻结全部{len(chosen)}个缺失月、{len(rows)}个周报自然刊期地址。",flush=True)


def collect() -> None:
    base.OUT=OUT
    nbg.OUT=OUT
    official=pd.read_csv(PARENT/"inputs/official_releases.csv")
    choices=pd.read_csv(OUT/"inputs/candidate_urls.csv")
    admitted,audit={},[]
    for variant in ("short_year","long_year","new_archive"):
        selected=choices[choices.variant.eq(variant)&~choices.stat_month.isin(admitted)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for rec in pool.map(base.fetch,selected.to_dict("records")):
                answer=base.qualify(rec,nbg.parse(rec["filename"]),official)
                answer["source_family"]="NBG_WEEKLY"
                audit.append(answer)
                if answer["status"]=="ADMITTED_RECONSTRUCTED_PRERELEASE_PAIR":
                    admitted[rec["stat_month"]]=answer
                print(f"{rec['stat_month']}：{answer['status']}，{answer['reasons']}",flush=True)
    base.save(OUT/"results/source_admission_audit.json",audit)
    base.save(OUT/"results/admitted_pairs.json",list(admitted.values()))
    base.save(OUT/"result.json",{"status":"SOURCE_SEARCH_COMPLETED_NO_RETURNS_READ","completed_at":base.now(),
        "admitted_pairs":len(admitted),"attempts":len(audit),"market_return_reads":0,"models_run":0,"accounts_run":0})
    print(f"补齐结束，新增{len(admitted)}个月。",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="NBG周报按自然刊期补齐全部缺失月")
    parser.add_argument("action",choices=["freeze","collect"])
    freeze() if parser.parse_args().action=="freeze" else collect()
