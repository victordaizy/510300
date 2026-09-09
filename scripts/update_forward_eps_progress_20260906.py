"""把用户明确的前瞻每股收益主线和真实已完成结果写入活动索引，保留旧账户计数。"""
from pathlib import Path
import hashlib
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now


def main():
    path=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
    d=read(path);before=path.read_bytes()
    snapshot=ROOT/"reports/research/510300_pre_forward_eps_progress_snapshot_20260906.json"
    if snapshot.exists():raise FileExistsError("前瞻主线进度更新已执行，禁止再次以旧快照覆盖")
    snapshot.write_bytes(before)
    delivery=read(ROOT/"deliverables/510300夏普1.2持续研究_前瞻EPS与金融盈利基础_GPT审阅_20260906.delivery.json")
    assert d["registered_configurations_in_this_resumption"]==161 and d["evaluation_accounts_in_this_resumption"]==342
    d["updated_at"]=now();d["status"]="TEN_ACCOUNT_ROUNDS_UNCHANGED_FORWARD_EPS_PRIMARY_FIVE_REPORT_PILOT_COMPLETE_CONTINUOUS_HISTORY_NEXT"
    d["research_primary_focus"]="用户明确主要看前瞻EPS；未来盈利预期及同年度修正、前瞻估值为主，历史财报用于输入和兑现检验。"
    d["goal_achieved"]=False
    d["latest_continuation_note"]="docs/510300_FORWARD_EPS_CONTINUATION_20260906.md"
    d["next_work"]="优先扩展免费公开券商研报的历史目录、原始年度EPS及口径和日期，构建同目标年度预测修正、机构覆盖/分歧、前瞻估值；当前接口相对年份不可按报告年使用，空值不填零，不回填今天的一致预期。逐机构保留版本，明确当年/下一年/未来十二个月，按历史沪深300成分与权重聚合，再登记有限候选并与价格、股东回报、公募需求、全期限流动性结合，实际滚动训练和完整成本账户。历史财报作为输入及标签校验，已完成的十五份年报与五份研报不重做。"
    d["running_studies"]=[x for x in d["running_studies"] if x["study"]!="510300_FINANCIAL_TTM_DEPENDENCIES_V1"]
    for r in d["running_studies"]:
        if r["study"]=="510300_ORIGINAL_EARNINGS_SOURCE_COMPLETION_V1":
            r["status"]="HISTORICAL_FINANCIAL_INPUTS_SUPPORT_FORWARD_EPS_RESEARCH"
            r["remaining_work"]="原24季报94事实及新增15年报同期59核心值、46附注记录已完成；连续一季报/三季报251原件保存、216初步主体通过、1连接失败。以支持前瞻预测输入与兑现校验为用途，历史滚动盈利不再作为主线替代。"
    d["running_studies"].append({"study":"510300_FORWARD_EPS_SOURCE_PILOT_V2","status":"FIVE_REPORT_PILOT_COMPLETE_CONTINUOUS_ANALYST_HISTORY_REQUIRED","path":"reports/research/510300_forward_eps_source_pilot_v2","original_reports":5,"forecast_eps_facts":15,"forecast_parent_profit_facts":15,"same_target_year_sampled_revision_links":8,"institutions":1,"companies":1,"is_market_consensus":False,"exact_next_twelve_month_eps_available":False,"api_year_mapping_evidence":"reports/research/510300_forward_eps_public_api_probe_v1/result.json","new_account_evaluations":0})
    d["running_studies"].append({"study":"510300_FINANCIAL_QUARTER_HISTORY_V1","status":"251_ORIGINALS_ARCHIVED_FACT_EXTRACTION_PENDING_SUPPORT_FORWARD_EPS","path":"reports/research/510300_financial_quarter_history_v1","selected_reports":252,"archived":251,"first_title_subject_passed":216,"latest_result":"reports/research/510300_financial_quarter_history_v1/transport_retry_result.json","pending_connection_failure":"1207543218","new_financial_facts_admitted":0,"new_account_evaluations":0})
    d["completed_source_rebuilds"]+=[{"study":"510300_FINANCIAL_ANNUAL_COMPONENTS_V1","result":"reports/research/510300_financial_annual_components_v1/result.json","documents":15,"core_facts":59,"ordinary_eps_component_records":46,"accounting_relations":75,"new_account_evaluations":0},
        {"study":"510300_FINANCIAL_TTM_VINTAGE_BRIDGE_V1","result":"reports/research/510300_financial_ttm_vintage_bridge_v1/result.json","anchors":12,"metric_bridges":36,"complete_original_arithmetic":34,"explicit_cny_original_arithmetic":31,"later_comparative_differences":6,"matched_quarter_bridges_with_cny":16,"new_account_evaluations":0}]
    d["new_evidence"]=[x for x in d["new_evidence"] if not x.startswith("新增15份全年和上年同期原始报告3863页")]
    d["new_evidence"] += ["用户最新明确前瞻每股收益为研究主线；历史财报用于输入与兑现检验。",
        "十五份年报同期报告新增59核心事实、46普通股EPS组成；与旧94事实合计153。75会计及EPS关系通过，16年报与8版本回归验证通过。",
        "十二锚点36指标中34原始运算完整、31人民币明确；6个后来比较数改变，16个年报分季桥接一致；未形成连续前瞻EPS。",
        "固定252份一季报三季报初次249份，一次正常重取后251份、4919页，216初步标题身份通过；1份连接失败，未新接纳财务因子。",
        "五份国信平安银行原始研报保存15年度EPS预测及15归母预测、8组同年度样本间修正；不是完整机构历史或市场一致预期。",
        "2024年研报的公开接口当前年份为2026，当年EPS字段2.67对应原件2026E而不是2024E；另两份接口EPS为空但原件有预测，历史字段必须按原件绝对年度处理。",
        "2025年研报落款3月15日而公开平台3月16日，需按较晚日期处理。原始日期元数据不等于当年不可变快照。"]
    d["deliveries"].append({"rounds":[],"stage":"前瞻每股收益原件与金融盈利基础","zip":delivery["zip_path"],"md":"deliverables/510300前瞻EPS研究_20260906/前瞻盈利_估值_股东回报与公募需求.md","bytes":delivery["bytes"],"sha256":delivery["sha256"],"members":delivery["members"],"new_account_evaluations":0})
    d["checks"]="本轮24项必要回归验证、包内保存数值只读核对及268原始PDF核对通过；完整包CRC、唯一成员、索引尺寸哈希及冻结引用通过。无新回测、无额外安全审计、未上传或声称外部审阅。"
    save(path,d)
    save(ROOT/"reports/research/510300_forward_eps_progress_update_20260906.json",{"updated_at":now(),"status":"FORWARD_EPS_USER_DIRECTION_AND_COMPLETED_SOURCE_PROGRESS_RECORDED","before_sha256":hashlib.sha256(before).hexdigest(),"after_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"account_counts_unchanged":True,"goal_achieved":False},exclusive=True)
    print("活动索引已转到前瞻盈利主线，十轮161配置342账户计数保留。")


if __name__=="__main__":main()
