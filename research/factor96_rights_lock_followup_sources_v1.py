"""取得已明确定位的四份原始披露；不扩展到收益或全市场重扫。"""
from copy import deepcopy
import json
import pandas as pd

from research.factor96_rights_lock_followup_v1 import OUT, ROOT, read, save, digest, now
import research.factor96_rights_issue_documents_v1 as collector

IDS=["1212493239","1221139139","1207134172","1202838760"]


def run():
    for f in read(OUT/"freeze.json")["files"]:assert digest(f["path"])==f["sha256"]
    frame=pd.read_parquet(ROOT/"reports/research/510300_factor96_issuance_catalogue_completion_v1/unique_documents.parquet")
    index={str(r["document_id"]):r for r in frame.to_dict("records")};targets=[]
    for rid in IDS:
        if rid in index:
            r=deepcopy(index[rid]);r["source_date_basis"]="既有巨潮目录日期"
        else:
            assert rid=="1202838760"
            r={"document_id":rid,"symbol":"600909.SH","title":"华安证券首次公开发行股票招股说明书摘要",
               "catalogue_date":"2016-11-22","catalogue_timestamp":"2016-11-22T00:00:00+08:00",
               "source_url":"https://static.cninfo.com.cn/finalpage/2016-11-22/1202838760.PDF",
               "source_date_basis":"巨潮正式披露URL日期，尚未独立核对目录回执"}
        r["review_role"]="HOLDING_IDENTITY_AND_QUANTITY_FOLLOWUP"
        r["historical_first_publication_verified"]=False;r["trading_feature_admitted"]=False
        targets.append(r)
    save(OUT/"targets.json",targets)
    save(OUT/"source_target_freeze.json",{"at":now(),"target_sha256":digest(OUT/"targets.json"),
         "queries":["中信证券 越秀 2022 配股 121480144 限售","华安证券 安徽国控 2019 12月 限售股 上市流通 2年 减持",
                    "华安证券 2019-11-30 首次公开发行限售股 上市流通 安徽国控 两年","华安证券 安徽国控 2021 两年 减持 承诺"],
         "domains":["cninfo.com.cn","sse.com.cn","hazq.com","yuexiucapital.com"],
         "limit":4,"reason":"H股结果含A股衍生锁定、后续解禁、华安原始解禁与IPO承诺；只保存所披露时点可得事实。"})
    collector.OUT=OUT
    rows=[]
    for target in targets:
        r=collector.extract(collector.download(target));rows.append(r)
        print(json.dumps({"公告":r['document_id'],"状态":r['status'],"页数":r.get('pages')},ensure_ascii=False),flush=True)
    save(OUT/"documents.json",rows)


if __name__=="__main__":run()
