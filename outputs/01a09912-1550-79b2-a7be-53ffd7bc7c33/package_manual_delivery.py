"""打包本次人工公式、原生核对与有界诊断，不复制其他历史研究包。"""
import csv
import hashlib
import io
import json
import shutil
import tomllib
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULT = ROOT / "reports/research/510300_daily_manual_signal_v1/2026-09-11"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, text):
    path = BASE / name
    path.write_text(text, encoding="utf-8")
    return path


latest = read(RESULT / "每日导出回执.json")
diagnostic = read(RESULT / "诊断结果.json")
native = read(BASE / "Excel原生重算核验.json")
auto = tomllib.loads(Path(r"E:\CodexData\.codex\automations\510300-1-2\automation.toml").read_text(encoding="utf-8"))
assert auto["status"] == "ACTIVE"
assert auto["rrule"] == "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=15;BYMINUTE=30;BYSECOND=0"
assert "research.daily_manual_signal_v1" in auto["prompt"]
assert native["status"] == "PASS_NATIVE_MICROSOFT_EXCEL_RECALCULATION"
workbook = BASE / "510300_D60每日手算.xlsx"
with zipfile.ZipFile(workbook) as z:
    assert z.testzip() is None
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    tree = ET.fromstring(z.read("xl/worksheets/sheet2.xml"))
    pane = tree.find("s:sheetViews/s:sheetView/s:pane", ns)
    assert pane is not None and float(pane.get("ySplit", "0")) >= 1, "数据表表头未冻结"
    assert float(pane.get("xSplit", "0")) >= 1, "日期列未冻结"
    assert "_xlfn.STDEV.S" in z.read("xl/worksheets/sheet2.xml").decode("utf-8")
snapshot = {"id":auto["id"],"name":auto["name"],"status":auto["status"],"rrule":auto["rrule"],
    "timezone":"Asia/Shanghai","target_thread_id":auto["target_thread_id"],"updated_at":auto["updated_at"],
    "next_expected_run":"2026-09-14T15:30:00+08:00","daily_export_module":"research.daily_manual_signal_v1",
    "note":"未来任务尚未运行；本地文件任务依赖电脑开机及应用运行。"}
write("每日任务设置回执.json", json.dumps(snapshot,ensure_ascii=False,indent=2))

main = [r for r in diagnostic["comparisons"] if r["period"] == "主历史"]
table = "\n".join(f'| {r["description"]} | {r["entry_condition_days"]} | {r["entry_changed_days"]} | {r["entry_changed_fraction"]:.2%} | {r["exit_changed_days"]} |' for r in main)
report = f"""# 510300 的差值、√60 与每日手算

截至2026年9月11日，现版 D60 = **1.062940770373**。9月10日为1.155144，因此连续两日大于1的入场因子条件成立，连续两日小于0的差值退出条件不成立。

完整组合 SELECTED_MIX_BAND10_SIMPLE2 的两档研究账户截至同日均持有0份，参考目标权重为0，针对9月14日的下一日净申请为0份。D60只是组合的上游条件，完整组合还有退出学习模型、来源账户状态、85/15混合和仓位限制。不能把本表的条件当成完整组合的最终买卖指令。这里全部是研究模型的保存决定，不是实际证券账户状态。

## 差值在衡量什么

不发生除息时：隔夜收益 n = LN(今天开盘/昨天实际收盘)，日内收益 i = LN(今天收盘/今天开盘)，每天的差值 d = i − n。

它衡量同一天日内相对隔夜的强弱。用对数收益，可使日内与隔夜的加法严格对应完整一日的对数财富收益。差值是在比较两段；相加则在衡量全天的总收益，研究问题已经不同。不能直接把差值叫作资金净流入或认定其因果机制成立。

现版含每份除息现金 q，精确口径为 n = LN((开盘+q)/昨收)，i = LN((收盘+q)/(开盘+q))。分红填在除息日；必须用不复权开盘与收盘、前一交易日实际收盘，不能再混用前复权序列。

9月11日实际例子：昨收4.617、开盘4.592、收盘4.579、分红0。隔夜对数收益−0.542948%，日内−0.283503%，差值却是+0.259446%。正差值表示日内跌得少于隔夜，并不表示当天上涨。

## √60 从哪里来

令 s 为最近60个日差值的样本标准差，则 F = Σd / (s × √60) = 平均d / (s / √60) = 平均d / s × √60。

加总形式中的√60在分母，平均值形式中的√60在分子。两式是恒等式，不会因表达方式不同而改变结果。均值的标准误在独立同分布假设下为 s/√N，因此这个分数在形式上类似检验均值是否为零的t统计量。该数学形式见[NIST的均值检验定义](https://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm)。

本策略并未因此取得有效的t检验显著性或收益概率。收益相关性、波动变化、滚动窗口重叠及历史筛选都会影响统计解释。√60不是年化系数。阈值1是策略规则，不是“95%置信”或“未来盈利概率”。

60是现版沿用的交易日窗口，大致对应一个季度，并非数学上唯一正确或已证明最优的长度。早期源码已枚举过多种窗口和方向，本次不根据新表格挑选新的最佳版本。

9月11日的60日差值和为0.093121448428，样本标准差为0.011310063207；0.093121448428/(0.011310063207×√60) = 1.062940770373。

## 换一下影响有多大

固定对照使用同一数据、同一连续两日判断。主历史为2020年至2026年9月11日，共1624个交易日。下表统计条件成立日期，不是实际交易次数或完整账户收益。较早历史对照也完整附在CSV中。

| 变动 | 入场条件成立日期数 | 相对现版入场条件不同日期数 | 占比 | 退出条件不同日期数 |
|---|---:|---:|---:|---:|
{table}

只去掉分母的√60，数值变成原来的7.7459667倍。把入场阈值同时由1改成√60，则本次所有日期的入场与退出条件完全恢复一致。把分母√60换成60，因子缩为原来的1/√60；阈值也相应改为1/√60时同样等价。零退出线不因正的尺度变换而改变。

20日、120日对照均保持源码的60日标准差估计窗口，只改累计期及对应√N。加法对照使用日内加隔夜序列自身的60日样本标准差。总体标准差对照将分数放大√(60/59)，约0.844%，在主历史仍造成5个入场条件日期变化。

这些是因子与门槛敏感性，未重跑27节点组合、账户成本或退出模型，没有得到公式变化后年化与夏普的比较。更频繁出现门槛不表示更高收益。

## 每天怎么填

打开《510300_D60每日手算.xlsx》，在“每日数据”第262行开始填写 A 日期、C 开盘价、D 收盘价、E 每份现金分红四项；非除息日E填数字0，不能留空。B昨收和F:N公式已预填至第1001行。

只录入交易日，按时间升序，不插空行、不漏交易日。收盘后用完整日线，Excel开启自动计算或按F9。返回“日常操作”即可看最新日期、分数、两日条件和下一条填写位置。预填260个交易日，最初59行建立窗口，后续201行分数与源码一致。

“每日数据”的列对应 A日期、B昨收、C开盘、D收盘、E每份现金分红、F隔夜、G日内、H差值、I60日差值和、J60日样本标准差、KD60、L入场条件、M差值退出条件、N输入状态。完整可复制公式见《Excel公式_可复制.txt》。Excel STDEV.S使用n−1样本标准差，见[Microsoft函数说明](https://support.microsoft.com/en-us/excel/functions/stdev-s-function)。

## 每日自动计算与本次验证

原任务“{auto['name']}”已更新并保持ACTIVE，安排为北京时间工作日15:30。按官方交易日判断是否有新的完整收盘，续算现有完整组合后导出同日D60与完整组合研究决定。下一预定运行是2026年9月14日15:30，尚未发生。需要本地文件的任务依赖电脑开机及应用运行，见[官方任务说明](https://learn.chatgpt.com/docs/automations?surface=app)。用户会手工续填的Excel不被定时任务覆盖。

全历史3476行、3416个有效D60分数已用标准库离线复算。Microsoft Excel {native['excel_version']} 原生重算201行有效分数及两日门槛，最大绝对误差{native['max_absolute_error']:.3g}。还检查续填、缺失分红、除息现金、严格大于阈值及零标准差。续填测试使用临时假设数据，关闭工作簿时未保存，未生成9月14日实际数据。

首次导出曾因标准差函数缺少OOXML兼容前缀导致原生Excel识别失败，已修正并重算通过。此前失败记录在证据目录保留。JS生成器在完成保存后返回非零退出码且没有抛出异常，未用该进程状态宣称生成成功；以保存文件检查、公式独立核对和原生Excel重算结果确认交付可用。

本包仅覆盖本次手工公式及因子敏感性，不重新收集市场数据，不复制上轮完整组合评审包，也不声称已验证稳定收益。
"""
write("01_差值与平方根60_说明及对照.md", report)
formulas = """510300 D60 每日手算公式（与交付Excel列映射一致）

数据从第2行开始，每行一个完整交易日。A日期，B前一交易日实际收盘，C开盘，D收盘，E每份除息现金。
价格必须不复权。非除息日E填数字0。分红在除息日录入，派息到账日不再重复录入。
现成Excel已含下列公式，不必重新粘贴。以下亦可用于从零建表。

B2：手填首行之前一个交易日的实际收盘价。
B3，向下填充：
=IF(A3="","",IF(ISNUMBER(D2),D2,""))

F2（隔夜对数收益），向下填充：
=IF(COUNT(B2:E2)<>4,"",IF(OR(B2<=0,C2<=0,D2<=0,E2<0),"",LN((C2+E2)/B2)))

G2（日内对数收益），向下填充：
=IF(COUNT(B2:E2)<>4,"",IF(OR(B2<=0,C2<=0,D2<=0,E2<0),"",LN((D2+E2)/(C2+E2))))

H2（日内减隔夜），向下填充：
=IF(COUNT(F2:G2)=2,G2-F2,"")

I61（最近60个交易日差值之和），向下填充：
=IF(COUNT(H2:H61)=60,SUM(H2:H61),"")

J61（最近60个交易日差值的样本标准差），向下填充：
=IF(COUNT(H2:H61)=60,STDEV.S(H2:H61),"")

K61（当前D60），向下填充：
=IF(COUNT(I61:J61)<>2,"",IF(J61=0,"",I61/(J61*SQRT(60))))

K61的等价写法（二者选其一，完全一样）：
=IF(COUNT(H2:H61)<>60,"",IF(STDEV.S(H2:H61)=0,"",AVERAGE(H2:H61)/STDEV.S(H2:H61)*SQRT(60)))

L62（连续两日严格大于1），向下填充。1为条件成立，0为不成立，空白为不可判断：
=IF(COUNT(K61:K62)<>2,"",IF(AND(K62>1,K61>1),1,0))

M62（连续两日严格小于0），向下填充：
=IF(COUNT(K61:K62)<>2,"",IF(AND(K62<0,K61<0),1,0))

I:J列不是价格收益率，I是对数差值的累计，J是该差值的样本标准差，K无量纲。
L/M仅是D60因子门槛。它们没有代替现版完整组合的退出模型、资金分配、持仓状态和实际可成交约束。
进入阈值1与退出阈值0都需要连续2日；等于阈值不算满足严格不等式。
历史前段必须有60个有效差值；这需要60天的开收盘及首日之前的昨收。无初始昨收时至少需要61天价格。

截至2026-09-11：D60=1.0629407703733325，L=1，M=0。
完整组合两档研究账户均持有0份、下一交易日净申请0份，不能只凭L=1就声称现版完整组合要求买入。
"""
write("Excel公式_可复制.txt", formulas)
write("00_README_FIRST.md", """# 510300 每日人工公式与敏感性对照

先打开 `510300_D60每日手算.xlsx`，在“每日数据”第262行的黄色输入格开始续填。只填日期、开盘、收盘、除息现金四项，其余公式自动计算。

再读 `01_差值与平方根60_说明及对照.md`。`Excel公式_可复制.txt` 可用于自己重新建表。

`数据/公式变化_门槛日期对照.csv` 包含主历史和较早历史的全部八种对照。日期数不等于交易次数。`数据/完整策略当前研究决定.csv` 是两档完整组合的研究决定，不是单个因子的门槛。

`verify_manual_csv.py` 可以用Python标准库离线复算同包全历史D60，不访问网络、不生成账户。`证据/` 保存公式、原生Excel、日更安排及失败修复记录。

这是本次公式研究的小包，不包含其他轮次的完整组合账户与网络原始响应，不代表外部GPT已经完成复核。
""")
write("02_GPT复核提示词.txt", """请复核本包510300人工公式与有界敏感性分析。重点检查：除息财富分解、日内减隔夜的经济含义、加总与均值两种√60表达的等价性、样本标准差、60个有效差值、严格不等式与连续两日确认、因子门槛与完整组合决定的区别、条件日期统计是否被夸大为收益证明。先运行可选的标准库离线复算，再检查Excel公式和原生重算回执。指出确切错误与证据路径。最后提出下一步优先顺序、最低验证标准、停止条件；不要在本包反复调参追逐更高历史成绩，也不要把少量新增日期或历史门槛变化当成稳定年化/夏普证据。
""")
write("03_用户请求.txt", "数据要每天算一下了，为什么是这个“差值”，为什么是*60平方根，这些换一下影响很大吗？写一个通信达或者exl的公式给我，我要人肉跑\n")
write("首次导出失败及修复.json", json.dumps({"initial_native_failure":"STDEV.S在OOXML中缺少_xlfn兼容前缀，Excel中J61报函数名称错误",
    "resolution":"用_xlfn.STDEV.S导出；Excel显示为常规STDEV.S，原生重算通过",
    "native_verification":native["status"],"builder_process_exit_code":1,
    "builder_note":"文件与预览已保存、无抛出异常；非零退出状态保留，不替代文件与原生重算核验",
    "initial_workbook_retained_locally":"failed_attempts/首次导出_STDEV_S原生重算失败.xlsx","failed_workbook_not_for_use":True},ensure_ascii=False,indent=2))

members = {p.name:p for p in [workbook,BASE/"00_README_FIRST.md",BASE/"01_差值与平方根60_说明及对照.md",BASE/"02_GPT复核提示词.txt",BASE/"03_用户请求.txt",BASE/"Excel公式_可复制.txt",BASE/"verify_manual_csv.py"]}
for filename in ["D60每日数据.csv","公式变化_门槛日期对照.csv","全部对照因子.csv","完整策略当前研究决定.csv","工作簿输入.json"]:
    members["数据/"+filename] = RESULT/filename
for filename in ["每日导出回执.json","诊断结果.json"]:
    members["证据/"+filename] = RESULT/filename
for filename in ["公式计算核对.json","公式错误扫描.json","Excel原生重算核验.json","CSV离线复算回执.json","每日任务设置回执.json","首次导出失败及修复.json"]:
    members["证据/"+filename] = BASE/filename
members["证据/公式对照预定方案.json"] = RESULT.parent/"公式对照预定方案.json"
for filename in ["research/daily_manual_signal_v1.py","research/simple_session_divergence_v1.py","research/intraday_overnight_increment_v1.py","config/510300_research_authority_v6.json","config/510300_new_daily_input_adapter_runtime_v1.json","data/reference/510300_dividends.csv"]:
    members["源码与输入口径/"+filename] = ROOT/filename
members["源码与输入口径/build_manual_workbook.mjs"] = BASE/"build_manual_workbook.mjs"
members["源码与输入口径/verify_excel_native.ps1"] = BASE/"verify_excel_native.ps1"

index_rows = [{"member":name,"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for name,p in sorted(members.items())]
buffer=io.StringIO(newline="")
writer=csv.DictWriter(buffer,fieldnames=["member","bytes","sha256"])
writer.writeheader()
writer.writerows(index_rows)
index_bytes=buffer.getvalue().encode("utf-8-sig")
(BASE/"FILE_INDEX.csv").write_bytes(index_bytes)
target=BASE/"510300_D60手算与公式诊断包.zip"
building=target.with_suffix(".building.zip")
with zipfile.ZipFile(building,"w",zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for name,p in sorted(members.items()):
        z.write(p,name)
    z.writestr("FILE_INDEX.csv",index_bytes)
with zipfile.ZipFile(building) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(index_rows)+1
    for row in index_rows:
        content=z.read(row["member"])
        assert len(content)==row["bytes"] and sha(content)==row["sha256"]
building.replace(target)
receipt={"status":"PASS_STRUCTURAL_ZIP_AND_NATIVE_EXCEL", "zip":str(target),"bytes":target.stat().st_size,
    "members":len(index_rows)+1,"indexed_members":len(index_rows),"sha256":sha(target.read_bytes()),
    "portfolio_returns_recomputed":False,"external_GPT_review":False,"network_requests_for_market_data":0}
write("交付包核验.json",json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
