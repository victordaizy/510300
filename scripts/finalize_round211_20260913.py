"""交付免费新行情覆盖、正式冻结时间边界及终点清仓恢复问题。"""
import json
import pandas as pd
from scripts.post_selection_data_feasibility_20260913 import CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    time_info=result['time_boundaries']
    require(result['price_comparison']['status']=='PASS_OHLC_AND_NEW_DATE_COVERAGE','价格覆盖结果不同')
    require(result['dividend_coverage']['status']=='OFFICIAL_CANDIDATE_COVERAGE_CONFIRMED_NO_LEDGER_CHANGE','分红核对结果不同')
    require(len(time_info['expected_gap_dates_before_strategy_freeze'])==20 and not time_info['expected_completed_dates_after_strategy_freeze'],'冻结前后日期范围不同')
    states=pd.read_csv(OUT/'terminal_state_inventory.csv')
    decision=('第211轮免费数据与实施前提检查完成。新浪和腾讯取得至9月11日的20个补充交易日，共同30日开高低收完全一致；'
        '官方分红候选覆盖核对到9月11日，无新增事件。20日都发生在209正式冻结之前，严格前瞻证据仍为零。发现原终点强制清仓不能直接用于自然续算，下一项处理输入和状态恢复。')
    detail=[f"本项零新候选、零新拟合、零账户、零新绩效计算，耗时{result['run_seconds']:.2f}秒。上面主历史和较早历史的指标只复用209简单方案作为上下文，不是使用新行情计算的成绩。四个来源各一次请求，总采购成本零。",
        '新浪与腾讯均有2026年8月3日至9月11日30条日线；8月17日至9月11日新增20个官方交易日，没有缺日期，120组开高低收值完全一致。成交量和成交额字段均非缺失，但本项没有完成两源单位与数值的进一步交叉核对，下一项接纳输入前补齐。',
        '官方基金登记仍为14个分红事件；上交所一次完整公告查询返回16条公告，已知分红锚点存在，没有未核对的新公司行为。已有分红账本没有修改；候选覆盖回执在本目录单独保存，未覆盖全局分红资料。',
        '209实际冻结时刻为2026年9月13日09:03:57。新取的二十个交易日都在该时刻之前，不能仅因为现在读取就称为项目从未见过或严格前瞻。下一官方日历交易日为9月14日，但尚未建立任何在执行前留存的新决定记录。',
        '简单方案主压力在8月13日收盘持有61800份、净值383490.35元。原规则给8月14日的正常申请为零，而终点开盘实际模拟卖出61800份；这是研究结算造成的清仓，不是自然信号。直接从清仓后的现金状态接续，会改变策略的真实持仓路径。',
        '', '## 保存来源与实际状态', '', '|来源|状态|原始响应|', '|---|---|---|']
    for row in result['requests']:
        detail.append(f"|{row['source']}|{row['status']}|{row.get('raw_file','无成功响应')}|")
    detail+=['','已保存来源为[新浪ETF日线](https://finance.sina.com.cn/realstock/company/sh510300/hisdata_klc2/klc_kl.js)、[腾讯日线接口](https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get)、[华泰柏瑞产品分红登记](https://www.huatai-pb.com/products/zhishu/510300/index.html)及[上交所公告查询](https://query.sse.com.cn/commonQuery.do)。具体请求参数、时间、服务器日期和原始字节哈希见本目录回执，单独打开无参数接口不等于相同查询。',
        '', '## 终点与正常决定盘点', '',
        '|账户|历史段|费用|终点前份额|终点前正常目标|正常申请|终点实际成交|终点后份额|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in states.to_dict('records'):
        detail.append(f"|{r['model']}|{'主历史' if r['period']=='evaluation' else '较早历史'}|{'基础' if r['cost']=='BASE' else '压力'}|{r['last_normal_shares']}|{r['source_normal_target']:.4%}|{r['source_normal_request']}|{r['terminal_actual_fill']}|{r['terminal_ending_shares']}|")
    detail+=['','## 下一项最短路径','','不再请求相同来源。先将已核对的二十个日期形成新目录候选输入，核对重叠价格、成交量额单位、日期与分红，并验证旧3456行因素前缀不变；随后只从简单方案两个非零来源追踪最小依赖和实际需要的状态。',
        '不能以日账本已经存在就假设内部持仓周期、等待计数、风险窗口和已成熟模型版本都可以直接恢复；也不能把终点强制清算当自然退出。必要的来源恢复入口和最小重放范围须在下一项落到具体文件，旧冻结数据、策略和历史通过结果全部保留。',
        '本项没有把新资料接入策略计算，没有生成新账户和新夏普，也没有启动券商、订单或Paper／Shadow。下一项规则已写定但尚未实现。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'new_candidates':0,'new_accounts':0,'goal_achieved':False,
        'price_coverage':'PASS','dividend_candidate_coverage':'PASS','strict_forward_evidence_days':0,
        'new_data_admitted_to_strategy':False,'status':'FREE_DATA_AVAILABLE_PRE_SELECTION_GAP_REQUIRES_STATE_RESTORATION'},exclusive=True)
    nxt=ROOT/'docs/510300_POST_SELECTION_EXTENSION_INPUTS_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300后续免费数据与时间边界_第211轮_20260913/后续数据可用性及连续状态恢复说明.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'后续免费数据与正式冻结时间边界',
        'COMPLETED_FREE_DATA_AVAILABLE_NO_FORWARD_DAYS_STATE_RESTORATION_REQUIRED',decision,'\n'.join(detail),
        nxt,nxt.read_text(encoding='utf-8').splitlines(),'POST_SELECTION_EXTENSION_INPUTS_PREPARED',
        '接纳已保存二十日补充行情并核对旧因子前缀，只追踪简单方案两个非零来源的最小依赖与恢复状态')
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=212,registered=False,planned_settings=0,planned_new_accounts=0,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False,
        research_class='FIXED_STRATEGY_EXTENSION_INPUTS_AND_MINIMAL_STATE_DEPENDENCIES',
        saved_source_result=str((OUT/'result.json').relative_to(ROOT)))
    index['status']='ROUND211_HISTORICAL_POINT_PASS_FREE_SUPPLEMENT_AVAILABLE_NO_STRICT_FORWARD_EVIDENCE'
    index['latest_post_selection_data_feasibility']=str((OUT/'result.json').relative_to(ROOT))
    index['latest_supplemental_price_cutoff']='2026-09-11'
    index['supplemental_price_days_after_original_cutoff']=20
    index['supplemental_data_accepted_into_strategy']=False
    index['strict_forward_evidence_days']=0
    index['independent_validation']='NOT_ESTABLISHED'
    write_json(path,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
