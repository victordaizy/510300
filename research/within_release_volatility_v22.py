"""同一货币发布周期内相邻周的风险过程、信息更新与剩余收益分解。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_within_release_volatility_v22"
EPS=1e-12
CUTOFF=pd.Timestamp("2026-09-11T21:00:00+08:00")
RELIEF="两点间D下降_最近5日同步缓和"
ROLL="两点间D下降_最近5日反而增强"
OTHER="两点间D未下降"
STATES=[RELIEF,ROLL,OTHER]
PHASES=["2018-2021","2022-2024","2025-2026"]
CHANNELS=["loan","tsf","orders","manufacturing","housing","banker","entrepreneur","profit"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def csv(name,frame):
    frame.to_csv(OUT/"results"/name,index=False,encoding="utf-8-sig",float_format="%.15g")


def freeze():
    if (OUT/"freeze.json").exists():
        raise RuntimeError("本轮范围已经冻结。")
    for name in ["inputs","results","figures","code"]:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    protocol={
      "at":now(),"study_id":"510300_WITHIN_RELEASE_VOLATILITY_V22",
      "previous_turn_classification":"PROGRESS：V21经营原表、库存机制和时钟已完成；完整目标仍未达到。",
      "question":"同一M1/M2发布周期中，后一个周观察的下行缓和，与已经发生的价格变化、原终点剩余空间、向后延长20日所得的新增尾段是什么关系？",
      "population":"保留原445周；仅在当时已有货币公告的103个周期内生成全部相邻周对，每对不得跨货币发布周期。最初6周无公告保持无周期。预计336对，数量由源表验证。不是事后挑每周期最佳、最低波或最后一周。",
      "contexts":"两个观察点各自保留原信贷、订单、社融、制造业成本、工业利润、估值、广度和来源时钟；追加V21同版本工业利润率及V20美国实际/名义利率主时钟、额外一天延迟。货币同一份不等于其他信息不变。政策目录不完整，零节点不是没有新闻。",
      "states":"主轴为两观察点D20平方尺度之差，零容差1e-12。下降时再按后点原最近5日下行平方和是否高于紧邻前5日区分；保持原5日定义，假期前后两观察点间隔另记，不把它们强称总是5天。其余未下降全部保留。不是预测规则。",
      "outcomes":"主20交易日E0、延迟一天E1均沿原起止日期。两点的原20日收益、未来风险都保留。还固定较早起点的原终点，计算较晚开盘到该终点的剩余毛收益；其日数较短，不冒充同持有期策略优势。",
      "return_bridge":"较早20日收益=等待至较晚开盘的原本金贡献+(较晚开盘价/较早开盘价)*固定旧终点剩余收益。较晚20日收益=固定旧终点剩余收益+新延长尾段贡献。二者差额再拆为负等待贡献、分母变化和延长尾段，三项精确相加。",
      "rights":"每个开盘入场当日除息不享分红；原持有人在较晚入场日应得分红划入等待段，后段按新持有人权益算。固定股数、分红留现金，非账户、未扣费用。等待至实际新开盘含后点收盘后跳空，不全称后点21点已知价格。",
      "known_path":"两个观察日收盘之间另算当时已发生的固定股数含分红收益；不把随后实际开盘价格或后续利率放进当时输入。",
      "summary":"每组按货币发布周期等权、周期内有效周对等权。旧前期2018至2021、旧后期2022至2024、新口径2025至2026分别披露；不挑最好阶段。原信用订单状态全列，不加新高低位或牛熊阈值。",
      "associations":"只描述两点D20变化与其间已发生收益、后点新20日收益、旧终点剩余收益、后点未来下行尺度的加权Spearman。无回归拟合，不称增量因果或独立预测检验。",
      "uncertainty":"关联的描述区间采用6个连续货币周期循环区块、2000次、seed=20260930；少于24个不同周期不算区间。多重比较未校正。原定义不随结果变化。",
      "overlap":"另按日期贪心取不重叠周对：每口径首个成熟对开始，下一个较早点入场严格晚于前对较晚20日退出；不搜相位。仍是已见历史的敏感性，不是独立验证。",
      "cases":"完整展示原2020-11、2020-12、2021-01、2024-08四个已见货币周期的全部周对，不按本轮收益选择其中某一对。",
      "limits":"宏观同周期只能消除该份货币读数差异，不能控制全部新消息、预期和政策。资料当前版本与时钟假设仍保留。统计模型和账户不重训，冻结失败不救援。",
      "new_models":0,"new_accounts":0,"orders_authorized":False,"independent_validation":False,"goal_achieved":False,
    }
    save("protocol.json",protocol)
    paths={
      "weekly.csv":ROOT/"reports/research/510300_macro_transmission_context_v4/results/445周_多层证据与原后续路径.csv",
      "market.csv":ROOT/"reports/research/510300_macro_volatility_observation_v2_run1/inputs/market_daily.csv",
      "industrial.csv":ROOT/"reports/research/510300_orders_cost_profit_bridge_v21/results/65期_成本费用与利润率同比分解.csv",
      "rates.csv":ROOT/"reports/research/510300_rate_context_generalization_v20/results/连续美国记录_同日利率原值与时钟.csv",
      "policy_nodes.csv":ROOT/"reports/research/510300_information_change_transmission_v6/inputs/policy_nodes.csv",
      "policy_clock_note.json":ROOT/"reports/research/510300_macro_transmission_context_v4/results/政策利率目录时钟补充.json",
    }
    receipts=[]
    for name,path in paths.items():
        target=OUT/"inputs"/name
        shutil.copy2(path,target)
        receipts.append({"file":name,"source":str(path.relative_to(ROOT)),"sha256":digest(target)})
    save("freeze.json",{"at":now(),"protocol_sha256":digest(OUT/"protocol.json"),"inputs":receipts})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print("第二十二轮已冻结：同一货币公告下的全部相邻周，不改原20日标签与经济状态。")


def wealth_path(market,i,j):
    part=market.iloc[i:j+1]
    dividends=part.dividend.to_numpy().copy()
    dividends[0]=0.
    return part.close.to_numpy()+np.cumsum(dividends)


def risk(market,i,j):
    wealth=wealth_path(market,i,j)
    entry=float(market.open.iloc[i])
    daily=wealth/np.r_[entry,wealth[:-1]]-1
    return {"return":wealth[-1]/entry-1,"worst":min(0.,float((wealth/entry-1).min())),
            "downside":float(np.sqrt(252*np.mean(np.minimum(daily,0)**2))),
            "rv":float(np.std(daily,ddof=1)*np.sqrt(252)) if len(daily)>1 else np.nan}


def bridge(market,locations,first,last,clock):
    f=f"{clock}_20_"
    out={"clock":clock,"status":"PENDING_OR_MISSING_ORIGINAL_WINDOW"}
    for name,row,suffix in [("a",first,"entry_date"),("b",first,"exit_date"),("c",last,"entry_date"),("e",last,"exit_date")]:
        out[name+"_date"]=row[f+suffix]
    if first[f+"status"]!="MATURE" or last[f+"status"]!="MATURE":
        return out
    a,b,c,e=[locations[out[name+"_date"]] for name in ["a","b","c","e"]]
    assert b-a+1==20 and e-c+1==20 and a<c<=b<e
    pa,pc,cb,ce=map(float,[market.open.iloc[a],market.open.iloc[c],market.close.iloc[b],market.close.iloc[e]])
    div_pre=float(market.dividend.iloc[a+1:c+1].sum())
    div_common=float(market.dividend.iloc[c+1:b+1].sum())
    div_tail=float(market.dividend.iloc[b+1:e+1].sum())
    wait=(pc+div_pre-pa)/pa
    common=(cb+div_common-pc)/pc
    tail=(ce-cb+div_tail)/pc
    scale=pc/pa
    original=risk(market,a,b)
    shifted=risk(market,c,e)
    common_risk=risk(market,c,b)
    assert abs(original["return"]-float(first[f+"return"]))<1e-11
    assert abs(shifted["return"]-float(last[f+"return"]))<1e-11
    assert abs(original["return"]-(wait+scale*common))<1e-11
    assert abs(shifted["return"]-(common+tail))<1e-11
    out.update(status="MATURE",a_index=a,b_index=b,c_index=c,e_index=e,waiting_sessions=c-a,
        common_sessions=b-c+1,tail_sessions=e-b,a_open=pa,c_open=pc,b_close=cb,e_close=ce,
        dividends_wait=div_pre,dividends_common=div_common,dividends_tail=div_tail,
        early20_return=original["return"],late20_return=shifted["return"],late20_worst=shifted["worst"],
        early20_downside=original["downside"],late20_downside=shifted["downside"],
        wait_contribution=wait,common_return=common,common_worst=common_risk["worst"],common_downside=common_risk["downside"],
        common_first_principal_contribution=scale*common,new_tail_contribution=tail,entry_price_scale=scale,
        shifted_minus_early=shifted["return"]-original["return"],removed_wait_component=-wait,
        denominator_component=(1-scale)*common,
        identity_early_error=original["return"]-wait-scale*common,
        identity_late_error=shifted["return"]-common-tail,
        identity_difference_error=(shifted["return"]-original["return"])-(-wait+(1-scale)*common+tail))
    return out


def enrich(weekly):
    extra=[]
    industrial=pd.read_csv(OUT/"inputs/industrial.csv").set_index("stat_month")
    rates=pd.read_csv(OUT/"inputs/rates.csv")
    clocks={key:pd.to_datetime(rates[f"{key}_known_at"]) for key in ["main","delay1"]}
    for row in weekly.itertuples(index=False):
        t0=pd.Timestamp(row.snapshot_at)
        add={"origin_id":row.origin_id}
        for clock,times in clocks.items():
            valid=rates[times.le(t0)]
            if len(valid):
                source=valid.iloc[-1]
                for key in ["date","nominal","real","compensation","nominal_delta20_bp","real_delta20_bp","compensation_delta20_bp",f"{clock}_known_at"]:
                    add[f"us_{clock}_{key}"]=source[key]
        add["industrial_margin_status"]=row.profit_status
        if row.profit_status=="AVAILABLE_RECONSTRUCTED" and row.profit_period in industrial.index:
            source=industrial.loc[row.profit_period]
            assert pd.Timestamp(source.known_at)<=t0
            for key in ["margin","margin_yoy_change","cost_per100","cost_yoy_change","fee_per100","fee_yoy_change","other_net_change_pp","known_at"]:
                add[f"industrial_{key}"]=source[key]
        extra.append(add)
    return weekly.merge(pd.DataFrame(extra),on="origin_id",validate="one_to_one")


def build():
    if (OUT/"results/build_receipt.json").exists():
        raise RuntimeError("结果已完成，不重复覆盖。")
    weekly=enrich(pd.read_csv(OUT/"inputs/weekly.csv"))
    market=pd.read_csv(OUT/"inputs/market.csv")
    locations={d:i for i,d in enumerate(market.date)}
    down=((market.close+market.dividend)/market.close.shift()-1).clip(upper=0).pow(2)
    csv("445周_原多层背景与利率利润补充.csv",weekly)
    frozen_pairs=[]
    for (regime,month),group in weekly.dropna(subset=["stat_month"]).groupby(["training_regime","stat_month"],sort=True):
        group=group.sort_values("observation_date")
        assert group.source_sha256.nunique()==1
        for key in ["m1_yoy_pp","m2_yoy_pp","spread_pp","delta3_spread_pp"]:
            assert group[key].nunique(dropna=False)==1,(month,key)
        for k in range(1,len(group)):
            frozen_pairs.append({"pair_id":group.iloc[k-1].origin_id+"__"+group.iloc[k].origin_id,
              "training_regime":regime,"stat_month":month,"early_origin_id":group.iloc[k-1].origin_id,
              "late_origin_id":group.iloc[k].origin_id,"within_cycle_order":k})
    assert len(frozen_pairs)==336
    csv("336对_只按日期确定的相邻周成员.csv",pd.DataFrame(frozen_pairs))
    indexed=weekly.set_index("origin_id",drop=False)
    fields=["observation_date","snapshot_at","past_return20","past_return60","v_rv20","v_downside20","v_upside20","v_down_fraction",
      "v_recent5_down_sum","v_previous5_down_sum","downside_window_state","internal_breadth20","joint_credit_orders_state",
      "delta3_spread_pp","d3_relative_current_log_pp","d3_relative_base_revision_log_pp",
      "loan_corporate_long_ytd_yoy_change_yi","loan_household_long_ytd_yoy_change_yi","loan_corporate_long_ytd_yoy_direction","loan_household_long_ytd_yoy_direction",
      "orders_period","orders_first_release_value","manufacturing_input_price_diffusion","manufacturing_output_price_diffusion",
      "profit_period","profit_profit_ytd_reported_yoy_pct","valuation_original_pe_official","bond_china_10y_yield",
      "industrial_margin_yoy_change","industrial_cost_yoy_change","industrial_fee_yoy_change",
      "us_main_real","us_main_nominal","us_main_compensation","us_delay1_real","us_delay1_nominal","us_delay1_compensation"]
    pairs,paths,updates=[],[],[]
    for membership in frozen_pairs:
        first=indexed.loc[membership["early_origin_id"]]
        last=indexed.loc[membership["late_origin_id"]]
        row=dict(membership)
        row["analysis_period"]="2018-2021" if membership["stat_month"][:4]<="2021" else ("2022-2024" if membership["stat_month"][:4]<="2024" else "2025-2026")
        for prefix,origin in [("early",first),("late",last)]:
            for field in fields:
                row[prefix+"_"+field]=origin[field]
        i,j=locations[first.observation_date],locations[last.observation_date]
        gap=j-i
        assert 0<gap<20
        row["snapshot_gap_sessions"]=gap
        row["observed_close_return"]=(market.close.iloc[j]+market.dividend.iloc[i+1:j+1].sum())/market.close.iloc[i]-1
        row["downside_change"]=last.v_downside20-first.v_downside20
        row["downside_square_change"]=last.v_downside20**2-first.v_downside20**2
        row["entered_down_sum"]=float(down.iloc[i+1:j+1].sum())
        row["exited_down_sum"]=float(down.iloc[i-19:j-19].sum())
        row["rolling_identity_error"]=row["downside_square_change"]-252/20*(row["entered_down_sum"]-row["exited_down_sum"])
        assert abs(row["rolling_identity_error"])<1e-11
        row["state"]=OTHER if row["downside_square_change"]>=-EPS else (ROLL if last.v_recent5_down_sum-last.v_previous5_down_sum>EPS else RELIEF)
        row["joint_state_changed"]=first.joint_credit_orders_state!=last.joint_credit_orders_state
        for channel in CHANNELS:
            before=first[channel+"_known_at"]
            after=last[channel+"_known_at"]
            changed=pd.notna(after) and (pd.isna(before) or pd.Timestamp(after)>pd.Timestamp(before))
            row[channel+"_source_updated"]=changed
            updates.append({"pair_id":row["pair_id"],"stat_month":row["stat_month"],"channel":channel,
              "early_known_at":before,"late_known_at":after,"newer_source_recorded":changed,
              "early_status":first[channel+"_status"],"late_status":last[channel+"_status"],
              "early_period":first[channel+"_period"],"late_period":last[channel+"_period"]})
        for clock in ["main","delay1"]:
            for typ in ["real","nominal","compensation"]:
                row[f"us_{clock}_{typ}_change_bp"]=100*(last[f"us_{clock}_{typ}"]-first[f"us_{clock}_{typ}"])
        for clock in ["E0","E1"]:
            path={**membership,"analysis_period":row["analysis_period"],"state":row["state"],**bridge(market,locations,first,last,clock)}
            paths.append(path)
        pairs.append(row)
    pairs=pd.DataFrame(pairs)
    paths=pd.DataFrame(paths)
    csv("336对_风险变化已实现价格与多层背景.csv",pairs)
    csv("672条_原终点剩余与延长尾段金额桥.csv",paths)
    csv("2688条_其他经营来源更新.csv",pd.DataFrame(updates))
    full=pairs.merge(paths[paths.clock.eq("E0")].drop(columns=["training_regime","stat_month","early_origin_id","late_origin_id","within_cycle_order","analysis_period","state"]),on="pair_id",validate="one_to_one")
    csv("全部周对_E0完整连接.csv",full)
    csv("原四个病例_全部相邻周.csv",full[full.stat_month.isin(["2020-11","2020-12","2021-01","2024-08"])])
    save("results/build_receipt.json",{"at":now(),"original_weeks":len(weekly),"macro_cycles":int(weekly.stat_month.nunique()),
      "unlinked_weeks":int(weekly.stat_month.isna().sum()),"adjacent_pairs":len(pairs),"return_bridges":len(paths),
      "mature_bridges":int(paths.status.eq("MATURE").sum()),"state_counts":pairs.state.value_counts().to_dict(),
      "max_rolling_identity_error":float(pairs.rolling_identity_error.abs().max()),
      "max_cash_identity_error":float(paths[["identity_early_error","identity_late_error","identity_difference_error"]].abs().max().max()),
      "goal_achieved":False})
    print(json.dumps(json.loads((OUT/"results/build_receipt.json").read_text("utf-8")),ensure_ascii=False,indent=2))


if __name__=="__main__":
    {"freeze":freeze,"build":build}[sys.argv[1]]()
