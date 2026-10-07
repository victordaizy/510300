"""固定36份已存配股上市公告，核对登记股份分类及额外持有期约束。"""
from collections import Counter
from pathlib import Path
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

PREVIOUS = ROOT / "reports/research/510300_factor96_rights_prospectus_v1"
OUT = ROOT / "reports/research/510300_factor96_rights_listing_split_v1"


def freeze():
    source = PREVIOUS / "combined_source_facts_with_restriction_addendum.json"
    facts = read(source)
    listings = [r for r in facts if r["role"] == "LISTING_NOTICE"]
    assert len(listings) == 36 and len({r["event_id"] for r in listings}) == 36
    save(OUT/"protocol.json", {"at":now(),"study_id":"510300_FACTOR96_RIGHTS_LISTING_SPLIT_V1",
        "previous_goal_turn":"PROGRESS_74_PROSPECTUS_COMPARISONS_AND_CHANGCHUN_CLASSIFICATION",
        "scope":"全部36份已存A股配股上市公告；35份补充股份分类复核、1份长春已完成案例作为复用对照。所有文件先固定，不依赖收益筛选。",
        "fields":["新增股份总数","登记无限售新增股数","登记限售新增股数","明确持有期承诺及适用对象","安排上市日","股份分类前后口径"],
        "quantity_rule":"优先明确新增分类数量；仅当表格明确同口径且无其他股本事件混入时，才可按前后之差计算。A/H、存量/增量、单位、股份登记分类和额外承诺分开。",
        "clock_rule":"只从该公告目录日末代理时点增加分类信息；实际配售结果不得提前提供上市分类；已有承诺不得因登记无限售自动解除。",
        "no_inference":"缺少数量保留未知；不设额外持有期不能推成无任何约束；尚无全部承诺版本时即时可卖量保持空。",
        "review":"机器候选与已读字段分别保存。原数量、版本冲突和不明确口径保留，不按后来的收益或结果解消。",
        "new_network_requests":0,"new_accounts":0,"new_returns":0,"new_models":0,
        "orders_authorized":False,"delivery_package_required":False})
    save(OUT/"targets.json",listings)
    save(OUT/"freeze.json",{"at":now(),"files":[{"path":str(p),"sha256":digest(p)} for p in
        (source,OUT/"protocol.json",OUT/"targets.json",PREVIOUS/"event_chains_with_restriction_addendum.json")]})
    print("固定36份上市公告，复用1份已核实案例，补充核对其余35份。")


def extract():
    for item in read(OUT/"freeze.json")["files"]:
        assert digest(item["path"])==item["sha256"]
    outputs=[]
    for row in read(OUT/"targets.json"):
        assert digest(row["text_path"])==row["text_sha256"]
        pages=read(row["text_path"])["pages"]
        candidates=[]
        for page in pages:
            text=normalized(page["text"])
            spans=[]
            for match in re.finditer(r"无限售|有限售|限售|锁定|持有期|新增上市|本次上市(?:流通)?的?股|股本结构",text):
                a,b=max(0,match.start()-100),min(len(text),match.end()+360)
                if spans and a<=spans[-1][1]:
                    spans[-1][1]=max(b,spans[-1][1])
                else:
                    spans.append([a,b])
            for a,b in spans:
                candidates.append({"document_id":row["document_id"],"page":page["page"],"start":a,"end":b,"literal":text[a:b],"reviewed":False})
        outputs.append({"document_id":row["document_id"],"symbol":row["symbol"],"event_id":row["event_id"],
            "known_at":row["known_at"],"pages":len(pages),"candidates":candidates,
            "existing_total_new_shares":row["updates"]["listing_announced_total_shares"],"trading_feature_admitted":False})
    save(OUT/"source_candidates.json",outputs)
    print("已定位",len(outputs),"份文件",sum(r["pages"] for r in outputs),"页，",sum(len(r["candidates"]) for r in outputs),"个合并片段。")
    print("候选片段数分布",dict(Counter(len(r["candidates"]) for r in outputs)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["freeze","extract"])
    args=parser.parse_args()
    freeze() if args.stage=="freeze" else extract()
