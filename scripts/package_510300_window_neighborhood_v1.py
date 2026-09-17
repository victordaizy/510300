"""保存局部窗口敏感性的解释、关键持仓日期及自包含复核包。"""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_session_window_neighborhood_v1"
DELIVERY=ROOT/"deliverables/510300_50_60_70日窗口敏感性_20260913"
DELIVERY.mkdir(parents=True,exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def report_table(rows):
    records=[]
    for n in [50,60,70]:
        b=next(r for r in rows if r["window"]==n and r["cost"]=="BASE")
        s=next(r for r in rows if r["window"]==n and r["cost"]=="STRESS")
        records.append(f'| {n} | {b["annual_return"]:.2%} | {b["sharpe"]:.3f} | {s["annual_return"]:.2%} | {s["sharpe"]:.3f} | {s["max_drawdown"]:.2%} |')
    return "| 窗口 | 基础年化 | 基础夏普 | 压力年化 | 压力夏普 | 压力最大回撤 |\n|---|---:|---:|---:|---:|---:|\n"+"\n".join(records)


protocol=read(OUT/"protocol.json")
result=read(OUT/"result.json")
assert result["new_accounts"]==110
episode=[]
for n in [50,60,70]:
    ledger=pd.read_csv(OUT/"master_accounts"/f"main_N{n}_STRESS"/"ledger.csv")
    part=ledger[ledger.date.between("2024-09-24","2024-10-09")].copy()
    part["window"]=n
    episode.append(part[["date","window","equity","shares","filled_quantity","net_return"]])
pd.concat(episode).to_csv(OUT/"2024_september_october_positions.csv",index=False,encoding="utf-8-sig")
report=f"""# 50、60、70日：现版对窗口变化敏感

把60日改成50日或70日，保持其他规则及现有退出模型系数不变，完整组合的历史表现明显下降。本次检验的是窗口附近的稳定性，不是寻找另一个历史最优数值。

## 对照定义

主对照：F_N = 最近N日(日内对数收益−隔夜对数收益)之和 / [同一N日差值的样本标准差 × √N]，N只取50、60、70。

累计窗口、标准差窗口与√N同步改变。连续两日大于1才满足入场因子，连续两日小于0满足差值退出条件。改变的是因子的回看窗口；最长持有60日、账户风险窗口30/60日、两日确认、止损追踪、85/15配比、仓位带和费用等保持原规则。两条退出模型沿用原来的月度系数及选择时点，不针对50/70日重新训练。

账号均为20万元起始研究账户，只有510300与人民币现金。基础费用为佣金万二、最低5元、单边滑点5bp；压力为佣金万四、最低5元、单边滑点10bp。242交易日年化、现金及无风险收益0、整百份及0.001价位、下一开盘执行、T+1及分红记账保持原程序。

## 主历史：2020-01-02至2026-09-11

三个窗口均含1624个交易日。60日直接使用已保存的正式连续账户；50/70日逐层重放全部27个必要节点、每版本22个账户，重算依赖账户的收益、预算、目标和持仓状态。

{report_table([r for r in result['metrics'] if r['period']=='main'])}

50日压力年化比60日降低约4.95个百分点，70日降低约6.08个百分点。两者均未达到原来年化10%、净夏普1.2的历史点值目标。60日在两个邻近窗口之间的表现优势很大，不能据此宣称60日附近是一段稳定的参数区域。

## 较早历史：2015-01-05至2019-12-31

为保持三个窗口可比，均以最后真实收盘计价，没有最后一天强制开盘清仓。该口径下各含1219个交易日。

{report_table([r for r in result['metrics'] if r['period']=='early'])}

此处60日与以前旧报告略有不同，原因已逐行核实：前1218行权益完全一致，只有最后一天旧版强制开盘清仓、新表按收盘持有估值不同。不是增加收益日期或改变费用。具体证据见early_baseline_clock_check.json。

## 为什么十天差别会改变这么多

主历史中，50日与60日因子的相关系数仍约0.935，70日与60日约0.944，但入场门槛分别有205天、177天不同，占1624天的12.62%、10.90%。条件经过退出模型、再入场许可、持仓周期及账户风险预算传递，最终参考目标在1625个决定日中分别有725天、789天不同。决定数比收益日期多1，是因为包含起始决定及最后一天对下一交易日的决定，并未多出一个收益样本。

2024年9月底提供一个具体例子：9月23日，50日分数为0.816492，60日为1.019305；9月24日分别为1.130238和1.251134。于是60日满足连续两日大于1，50日没有满足。

保存的压力主账户显示：50日9月25日至10月8日全程空仓；60日9月25日买入68300份，并在10月8日开盘退出；70日9月25日增加至55100份，9月27日已全部退出。这些是完整组合的实际研究账本差异，不能只看因子曲线相关性。

整个2024年的压力账户收益分别为：50日−4.13%，60日+26.55%，70日+3.39%。完整年度表同时保留其他年份；这段行情解释了相当一部分差异，并非断言所有差异只由这一段造成。

## 结论的范围

这次结果支持：在现有模型系数及其他规则固定的条件下，60日改50/70日会明显影响绩效，局部窗口稳健性不足。它不能单独证明策略完全无效，也没有回答重新训练整个50/70日体系后的结果。各窗口均使用已经研究过的历史，不能被称为新独立验证。

原始代码还存在另一种字面改法：只改变累计窗口及√N，标准差仍使用60日。本包另保留该口径的因子对照STD_FIXED_60，但没有把它的数字混进上述完整账户表。

如后续研究继续，优先追溯关键周期的门槛、退出模型与仓位放大如何共同产生对60日的依赖。若要重训50/70日，须作为完整训练链的另一项事前固定对照，不用反复搜索55/58/62等数值解释掉当前失败。停止条件是预定邻近窗口继续依赖单个盈利周期、或成本后结果只在单点参数上漂亮时，不提升为已经验证的稳定策略。

## 文件与核对

本轮新增5份完整依赖图、110条研究账户，核对{result['verified_ledger_rows']}条账本的财富恒等式、费用后收益及盈亏分解。两条保存模型未重拟合，日常运行版本和手算Excel未改变，严格前瞻新增日期为0。

数据中的master_accounts包含三窗口、两时期、两费用共12份完整主账户CSV。完整内部账户、全部因素与状态保留在工作区快照。verify_510300_window_neighborhood_saved.py仅用Python标准库即可复算包内因子、12份主账户及其绩效。包装核对与经济复算均不代表外部GPT已经完成评审。
"""
(DELIVERY/"01_50_60_70日窗口对照报告.md").write_text(report,encoding="utf-8")
(DELIVERY/"00_README_FIRST.md").write_text("# 阅读顺序\n\n先读01_50_60_70日窗口对照报告.md。数据/portfolio_metrics.csv是完整组合比较，数据/signal_comparison.csv是因子门槛比较，二者不要混淆。数据/master_accounts保存12份主账户。工作区快照保存完整内部账户、输入、模型、配置及源代码。\n\n主比较同步改变累计、标准差和平方根窗口，其余规则和已存模型固定。主历史60日复用当前正式账户；较早历史统一按最后收盘估值。核对脚本默认读取同目录的数据文件夹，离线运行不新增账户或模型。\n",encoding="utf-8")
(DELIVERY/"02_GPT复核提示词.txt").write_text("请复核50/60/70日窗口局部敏感性：核实只有因子回看窗口改变，完整27节点目标与持仓是否真正重算，原模型系数保持固定的含义与局限，两个时期的末日时钟、费用和年化是否一致。运行标准库离线核对，检查2024年9月底的逐日条件与持仓证据。说明结论应如何改变对现版稳健性的判断，并给出下一项最小必要验证、优先级和停止条件。不要将历史窗口再优化当独立验证，不授权任何券商、订单或实盘动作。\n",encoding="utf-8")
(DELIVERY/"03_用户请求.txt").write_text(protocol["user_request"]+"\n",encoding="utf-8")

members={p.name:p for p in DELIVERY.glob("*.md")}
members.update({p.name:p for p in DELIVERY.glob("*.txt")})
for name in ["portfolio_metrics.csv","signal_comparison.csv","all_factor_values.csv","signal_inputs.csv","portfolio_decision_changes.csv","yearly_returns.csv","2024_september_october_positions.csv"]:
    members["数据/"+name]=OUT/name
for p in (OUT/"master_accounts").rglob("*.csv"):
    members["数据/"+str(p.relative_to(OUT)).replace("\\","/")]=p
for p in OUT.rglob("*"):
    if p.is_file() and "master_accounts" not in p.parts:
        members["工作区快照/"+str(p.relative_to(ROOT)).replace("\\","/")]=p
for item in protocol["frozen_files"]:
    p=ROOT/item["path"]
    assert digest(p.read_bytes())==item["sha256"],p
    members["工作区快照/"+str(p.relative_to(ROOT)).replace("\\","/")]=p
members["工作区快照/config/510300_research_authority_v6.json"]=ROOT/"config/510300_research_authority_v6.json"
members["verify_510300_window_neighborhood_saved.py"]=ROOT/"scripts/verify_510300_window_neighborhood_saved.py"
members["工作区快照/scripts/package_510300_window_neighborhood_v1.py"]=Path(__file__)

# 根据实际导入关系收集本地研究模块，不扫描大型数据目录。
queue=[ROOT/"research/session_window_neighborhood_v1.py"]
visited=set()
while queue:
    p=queue.pop()
    if p in visited or not p.is_file():
        continue
    visited.add(p)
    members["工作区快照/"+str(p.relative_to(ROOT)).replace("\\","/")]=p
    tree=ast.parse(p.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        modules=[]
        if isinstance(node,ast.ImportFrom) and node.module:
            modules=[node.module]
        elif isinstance(node,ast.Import):
            modules=[n.name for n in node.names]
        for module in modules:
            if module.startswith("research."):
                queue.append(ROOT/Path(*module.split(".")).with_suffix(".py"))

rows=[{"member":name,"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())} for name,p in sorted(members.items())]
buffer=io.StringIO(newline="")
writer=csv.DictWriter(buffer,fieldnames=["member","bytes","sha256"])
writer.writeheader();writer.writerows(rows)
index=buffer.getvalue().encode("utf-8-sig")
(DELIVERY/"FILE_INDEX.csv").write_bytes(index)
target=DELIVERY/"510300_50_60_70日窗口敏感性_GPT复核包.zip"
temporary=target.with_suffix(".building.zip")
with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(members.items()):
        z.write(p,name)
    z.writestr("FILE_INDEX.csv",index)
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))==len(rows)+1
    for row in rows:
        content=z.read(row["member"])
        assert len(content)==row["bytes"] and digest(content)==row["sha256"]
temporary.replace(target)
receipt={"status":"PASS_STRUCTURAL_ZIP","zip":str(target),"bytes":target.stat().st_size,"members":len(rows)+1,
    "indexed_members":len(rows),"sha256":digest(target.read_bytes()),"included_research_modules":len(visited),
    "security_audit":False,"external_GPT_review":False,"fresh_full_graph_replay_from_zip":False}
(DELIVERY/"ZIP结构核验回执.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False))
