"""检验混合类型年度表头和精确预测值可逆保存。"""
import json
import pandas as pd
from research.forward_eps_guosen_history_v1_1 import parquet_frame


def test_header_and_exact_eps_parquet_roundtrip(tmp_path):
    originals=[{'header':[[2022,'A'],[2023,''],[2024,'E']],
                'eps_value_exact':'-0.0100','target_fiscal_year':2024,'share_snapshot_million':None},
               {'header':[[2023,'A'],[2024,'E'],[2025,'E']],
                'eps_value_exact':'2.670000','target_fiscal_year':2025,'share_snapshot_million':'19406'}]
    frame=parquet_frame(originals)
    path=tmp_path/'facts.parquet';frame.to_parquet(path,index=False)
    saved=pd.read_parquet(path).to_dict('records')
    assert len(saved)==len(originals)
    for a,b in zip(originals,saved):
        assert a['header']==json.loads(b['header_json'])
        assert a['eps_value_exact']==b['eps_value_exact']
        assert a['target_fiscal_year']==b['target_fiscal_year']
