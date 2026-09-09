"""将第十轮完整结果与原始申赎来源进展接入持续研究索引。"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.intraday_overnight_increment_v1 import now,require,write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    path=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
    state=read(path)
    require(len(state["completed_rounds"])==9 and state["registered_configurations_in_this_resumption"]==154,"最新索引已变化，不覆盖")
    folder=ROOT/"reports/research/510300_original_quarterly_flow_policy_v1"
    result=read(folder/"result.json")
    receipt=read(ROOT/"deliverables/510300夏普1.2持续研究_第十轮原始申赎_GPT审阅_20260906.delivery.json")
    require(receipt["portable_saved_verifier_from_standalone_delivery"]=="PASS","独立交付目录复算未完成")
    require(hashlib.sha256(Path(receipt["zip"]).read_bytes()).hexdigest()==receipt["sha256"],"第十轮交付包哈希变化")
    write_json(ROOT/"reports/research/510300_sharpe_1_2_pre_round10_index_snapshot.json",state,exclusive=True)
    base,stress=[next(x for x in result["primary"] if x["cost"]==cost) for cost in ["BASE","STRESS"]]
    state["completed_rounds"].append({"round":10,"study":result["study_id"],"title":"原始公告季度申购赎回与价格增量",
        "status":result["status"],"result":"reports/research/510300_original_quarterly_flow_policy_v1/result.json",
        "candidate_configurations":7,"evaluation_accounts":16,"primary_base":base,"primary_stress":stress,
        "post_selected_best_base":result["post_selected_best_base"],
        "source_coverage_note":"56份原始季度PDF全部份额提取、55份官方时钟、55次季度衔接；53个合格季度起点，评价28事件、84次拟合。单只ETF份额不是全市场公募现金。"})
    configurations=sum(r["candidate_configurations"] for r in state["completed_rounds"])
    accounts=sum(r["evaluation_accounts"] for r in state["completed_rounds"])
    require(configurations==161 and accounts==342,"十轮配置或账户合计不符")
    allframes=[]
    for row in state["completed_rounds"]:
        f=pd.read_csv((ROOT/row["result"]).parent/"metrics.csv")
        f.insert(0,"研究名称",row["title"])
        f.insert(0,"轮次",row["round"])
        allframes.append(f)
    combined=pd.concat(allframes,ignore_index=True)
    require(len(combined)==342 and not combined.meets_point_target.any(),"全期目标状态需要重新判断")
    best=combined.loc[(combined.cost=="BASE") & ~combined.model.isin(["BUY_HOLD","CASH"])].sort_values("net_sharpe",ascending=False).iloc[0]
    require(best.model=="K2_MONTH_EDGE" and abs(best.net_sharpe-.5196477273626268)<1e-12,"最高结果发生变化，需要更新结论")
    state.update({"updated_at":now(),"goal_achieved":False,
                  "status":"TEN_ROUNDS_COMPLETED_ORIGINAL_QUARTERLY_FLOW_NOT_SUFFICIENT_FUNDAMENTAL_SOURCE_RESEARCH_CONTINUES",
                  "registered_configurations_in_this_resumption":configurations,"evaluation_accounts_in_this_resumption":accounts,
                  "latest_continuation_note":"docs/510300_ROUND10_CONTINUATION_20260906.md","automation_status":"ACTIVE"})
    state["running_studies"]=[r for r in state["running_studies"] if r["study"]!="510300_ORIGINAL_FUND_SUBSCRIPTION_REPORTS_V1"]
    state.setdefault("completed_source_rebuilds",[]).append({"study":"510300_ORIGINAL_FUND_SUBSCRIPTION_REPORTS_V1",
        "status":"HISTORICAL_56_QUARTERS_COMPLETED_55_OFFICIAL_CLOCKS_ONE_EXCLUDED",
        "result":"reports/research/510300_original_fund_subscription_reports_v1/quarterly_facts_v1_1_result.json",
        "original_quarterly_pdfs":56,"official_publication_clocks":55,"quarter_transitions_verified":55,
        "excluded_period":"2013Q2","used_in_new_study":"510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1"})
    state["deliveries"].append({"rounds":[10],"zip":"deliverables/510300夏普1.2持续研究_第十轮原始申赎_GPT审阅_20260906.zip",
        "md":"deliverables/510300第十轮_原始季度申购赎回_20260906/第十轮结果与全部中文因子规则.md",
        "bytes":receipt["bytes"],"sha256":receipt["sha256"],"members":receipt["members"]})
    state["new_evidence"].extend([
        "第十轮7候选16账户全部未达夏普1.2，主方案负0.013532，反向申赎事后最高0.311400。",
        "恢复83条上交所定期目录及56份季报；第一版9份跨页表不足已在收益前新版本修正，56份数值及55次季度衔接全部吻合。",
        "55份官方公布日期可用，2013Q2没有官方目录时钟不能进策略；28个评价事件全部有效，25至51份成熟季度支持84次拟合。"])
    state["next_work"]="继续原始盈利、估值、股东回报与全市场公募需求。第十轮单只ETF原始季度申赎已完成并交付，不能无限调整其方向、阈值或窗口挽救结果。优先扩展24份金融报告中尚未支持的22份，农业银行2019Q3及2021Q3的合并利润与EPS分列相邻两页、表头区分3个月及9个月；平安银行营业支出未含独立列示减值，不能简单照搬其他银行营业收入加营业支出等式。按原始证据登记新适配器，补普通股盈利、股数、优先股及其他权益分配扣除。全市场77份协会月报原始发布日期及分类仍需修复。EPS乘PE是价格分解，不能循环倒算做新预测；分红和回购避免重复计回报。继续固定有限新机制并验证完整成本账户，不改变旧冻结研究。"
    write_json(path,state)
    combined.to_csv(ROOT/"deliverables/510300夏普1.2持续研究_十轮完整指标_20260906.csv",index=False,encoding="utf-8-sig")
    lines=["# 夏普1.2持续研究：截至第十轮","","累计161个登记配置、342条完整评价账户，包含重复对照。所有候选全期仍未达到夏普1.2，独立高夏普与稳定超额证据未建立，目标保持未完成。","",
           "| 轮次 | 研究 | 预设主方案基础夏普 | 主方案压力夏普 | 本轮事后最高夏普 | 候选数 | 完整账户数 |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r in state["completed_rounds"]:
        lines.append(f"| {r['round']} | {r['title']} | {r['primary_base']['net_sharpe']:.4f} | {r['primary_stress']['net_sharpe']:.4f} | {r['post_selected_best_base']['net_sharpe']:.4f} | {r['candidate_configurations']} | {r['evaluation_accounts']} |")
    lines += ["","总体事后最高仍是第六轮月末月初规则0.5196。第十轮主方案年化负1.51%、最大回撤48.11%；本轮最高净赎回反向规则夏普0.3114、年化2.63%、回撤11.55%，年化收益低于买入持有3.63%。前后分期与不确定性均完整披露，不能挑好时期宣布完成。","",
              "原始申赎资料已完成2012Q3至2026Q2共56份PDF，55份官方公布时钟，全部份额等式和55次季度衔接通过。9个跨页问题在策略收益读取前修正，旧版记录保留。2013Q2公布时钟缺失、2013Q3缺合格上季信息、2012Q3缺上季，均保留不可用状态。","",
              "第十轮规则只在真实公告后的交易日及一次初始化判断，下一开盘执行。每份季报只占一个训练起点；28评价事件、84次三组匹配模型拟合均已保存。原始份额不是精确人民币流量，单只ETF不能替代全市场公募。","",
              "继续补金融成分公司合并盈利、普通股EPS、股本、分红与回购实际执行，以及全市场基金申赎原始来源。已有账户包含分红，新增股息因子不再重复加同一现金；实际回购对股数与EPS的影响不再重复计收益。","",
              "第十轮中文报告和盈利估值框架位于deliverables/510300第十轮_原始季度申购赎回_20260906。审阅ZIP含56份原始PDF、目录、修正前后事实、全部新结果、账户、模型、冻结规则和完整文件索引。包大小27,649,899字节、492成员；结构与保存结果复算通过，未上传或收到外部审阅。"]
    (ROOT/"deliverables/510300夏普1.2持续研究_截至第十轮_20260906.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    note="""# 第十轮后的继续执行记录

最新索引reports/research/510300_sharpe_1_2_latest_research.json优先。目标active：成本后完整账户夏普至少1.2及稳定超额，未完成，不标complete或blocked。用户明确不完成不停止、只用免费来源。实际研究交易范围510300与现金，没有其他ETF新授权、券商或真实订单权限。不新建任务，不擅自派子代理。每轮中文规则和自包含GPT审阅ZIP，只作必要结构数值核对，减少无关审计。

第十轮510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1已一次完成并交付，不重跑、不调阈值、不覆盖。冻结清单config/510300_original_quarterly_flow_policy_v1_manifest.json，SHA256为9c5a13c790e355c3f853128ae29d7c3cdab3a38af44a524734a582e5d5e688a2。7候选16评价账户，完整2020-01-02至2026-08-14开盘1604日；累计十轮161配置342账户均未达1.2，总体最好仍第六轮0.519647727。原始PDF及价格资料已有很多历史观察，不是独立验证。

第十轮主方案M1_PRICE_FLOW基础夏普负0.013532100、压力负0.020844615，年化负1.5081%、回撤48.1101%，19次基础成交。仅价格对照夏普0.010739，仅申赎0.136865；本轮最高R2_NEGATIVE_FLOW夏普0.311400413，年化2.6272%、回撤11.5527%，低于买入持有3.6257%年化。主方案夏普区间[-0.752125,0.682248]，相对仅价格年化日均增量区间[-0.0279415,0.0229477]，不能确认申赎增加稳定信息。

因子：mom60、sma120、vol60，季报净申赎除期初份额、该比率较上季变化、申加赎除期初份额。三组岭回归价格加份额、仅价格、仅份额，同一批六因子有效的季度样本，标准化仅训练集、alpha10、H60、最少12成熟季度、扩展历史。四规则顺向、反向、季度趋势、正申购且趋势。每份季报只一个训练起点，严格标签退出<=拟合日，不重复季度值成每日训练观察。28评价事件、84模型，实际训练25至51成熟季度。只在公告后首个交易日收盘及2019-12-31初始化决策，次开盘执行。初始化用2019Q3原始报告。新信息缺失不退旧资料，无判断保持实际份额，不能改成清仓；非事件保持份额。新research/event_clock_account_v1.py与原账户基准逐日一致；源代码已冻结。

原始申赎来源reports/research/510300_original_fund_subscription_reports_v1已完成56份2012Q3至2026Q2PDF、55个官方目录日期、全部56份数值和55次相邻季度衔接。原始RAW在E:/ResearchData/New project 8/data/raw/510300_original_fund_subscription_reports_v1。上交所公开接口query.sse.com.cn/commonQuery.do，sqlId COMMON_PL_JJXX_JJGG_NEW_L，SECURITY_CODE510300，BULLETIN_TYPE=reits03,fund03；参数证据在官方前端sse_search_fund.js。按2012至2026年查，83条定期公告，15个年度查询都一页，声明计数和唯一URL一致。55份SSE季报加2013Q2巨潮原始62852479.PDF。早期三份短标题经封面年份季报代码确认，原52份分类与缺失记录保留。标题分类脚本reconcile_original_fund_quarterly_catalog_20260906.py已冻结，不再改。

原始sourceparser research/original_fund_quarterly_facts_v1.py第一次47份通过，9份跨页未读完整；在收益前登记v1_1相邻页修正，全部56数值成功，原47记录不动。2013Q2无官方公布目录时钟，仍NO_VIEW；2013Q3缺合格上季，2012Q3缺上季，本轮53合格季度起点。可用时钟取公告日与封面送出较晚，再下一A交易日；PDF元数据只排查晚生成，不能证明公布。早期PDF属性Title用了其他基金旧模板，身份据正文封面与基金代码。对应manifest、首次失败及修正输出全部保存。10项针对性测试通过；只读核验84保存模型、16账户、三项价格因子和原始份额、时钟、区间通过，最大指标误差4.547e-13。解压交付目录独立复算也通过，无新拟合/账户/抽样/下载。

第十轮ZIP deliverables/510300夏普1.2持续研究_第十轮原始申赎_GPT审阅_20260906.zip，27,649,899字节，492成员、491索引行、204冻结文件；SHA256 a55ec52b38ae0b979b69298b42325a86748d7e2a0c8ad241d3f8800ec95613d2。两幅图已查看。中文说明、经济框架、阅读顺序、GPT评议提示词、完整PDF/目录/原始事实/全部新结果与账户/代码测试/文件索引齐备。前九轮仅保存指标摘要作背景，完整旧证据在旧包；没有再次扩展旧ZIP。不重打包已完成第十轮。结构验证不等于外部审阅或安全审计，未上传。

优先接下来实际补金融原始来源，当前只有2份招商银行版式、8事实通过的旧适配器。原始24份报告在reports/research/510300_original_earnings_source_completion_v1/batch_01_result.json。第一批12公司24份，另22份待扩展，有证据再登记新版本。农业银行2019Q3原始1207021230第18页合并利润、第19页合并续表EPS；2021Q3原始1211420908第17页合并利润、第18页EPS。相同表头“截至9月30日止3个月/9个月”，四列当年上年当年上年，累计为第3列。文本初查2019收入474981百万元、营业利润216693、归母180671、EPS0.51；2021收入544897、营业利润230946、归母186709、EPS0.50。尚未建立这个新适配器或正式准入，先视觉及表头/会计等式核对，再冻结规则。不能拿本期比较列充当上年原始披露。

平安银行2023Q3原始1218135621第35页“营业支出”不包含后面另列信用减值、其他资产减值。收入加营业支出等于“减值损失前营业利润”，还要减值才能得营业利润；直接照搬其他银行等式会错误拒绝真实报告或选错项目。这是文本新发现，尚未完成适配器。继续区分合并/母公司、累计/单季、普通股盈利与其他权益工具，不能把旧非金融排除金融的研究说成全错。初始2686份缺口193公司，先完成现有24份版式再推进完整历史。EPS、PE与股数必须同口径，回购注销对EPS与现金分红不重复计回报。

全市场公募仍从reports/research/510300_fundamental_and_fund_flow_rebuild_v1续行。77份AMAC月报154股票/混合行，44早期PDF元数据2023-11-21、25旧目录日疑似迁移、5相邻同范围份额差异及2025年11月分类变化。原始可用时钟未证明，不进策略。不要把单只ETF季报完成称全部公募已完成。可找同期官方独立发布与原始版本；净值规模变化不等于申购金额。

第七轮全期限逆回购已完成；各期限公开操作量、买断式月报与预告分开，不叫实际每日净投放，不退回7天单期限。九轮旧失败归因和结果详见上一续行说明。已观察历史不做参数救援，但用户授权新机制与有实质证据的来源修正版，不能仅做审计停原地。保持有限研究、完整账户、原始时钟、明确缺失和所有结果披露。每次实际推进不依赖阻塞源的独立工作，目标仍active。旧索引更新脚本均不得覆盖第十轮。
"""
    (ROOT/"docs/510300_ROUND10_CONTINUATION_20260906.md").write_text(note,encoding="utf-8")
    write_json(ROOT/"reports/research/510300_sharpe_1_2_round10_continuation_receipt.json",
               {"recorded_at":now(),"status":state["status"],"configurations":161,"evaluation_accounts":342,
                "goal_achieved":False,"zip_sha256":receipt["sha256"],"continued_sources":[r["study"] for r in state["running_studies"]]},exclusive=True)
    print(json.dumps({"状态":state["status"],"登记配置":161,"完整账户":342,"目标已完成":False},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
