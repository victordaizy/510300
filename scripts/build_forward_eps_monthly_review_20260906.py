"""交付金融前瞻盈利完整诊断，附带历史沪深300扩展的冻结规则与目录。"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from scripts.verify_forward_eps_monthly_saved_20260906 import verify

DELIVERY=ROOT/'deliverables/510300前瞻EPS月度诊断_20260906'
ZIP=ROOT/'deliverables/510300夏普1.2持续研究_前瞻EPS金融诊断与沪深300扩展_GPT审阅_20260906.zip'
REPORT='前瞻EPS策略_金融诊断结果与下一步.md'
FOLDERS=['510300_forward_eps_history_directory_v1','510300_forward_eps_csi_directory_v1',
         '510300_forward_eps_financial_originals_v1','510300_forward_eps_financial_facts_v1',
         '510300_forward_eps_financial_facts_v1_1','510300_forward_eps_monthly_policy_v1_financial']
MANIFESTS=['510300_forward_eps_history_directory_v1','510300_forward_eps_csi_directory_v1',
           '510300_forward_eps_financial_originals_v1','510300_forward_eps_csi_originals_v1',
           '510300_forward_eps_financial_facts_v1','510300_forward_eps_financial_facts_v1_1',
           '510300_forward_eps_csi_facts_v1','510300_forward_eps_csi_facts_v1_1','510300_forward_eps_monthly_policy_v1']


def prepare():
    if (DELIVERY/REPORT).exists():raise FileExistsError('本轮中文说明已经生成')
    DELIVERY.mkdir(parents=True,exist_ok=True)
    b=ROOT/'reports/research';out=b/'510300_forward_eps_monthly_policy_v1_financial'
    result=read(out/'result.json')
    names={'E1_PRICE_FORWARD_EPS':'主方案：价格加前瞻盈利','E2_MATCHED_PRICE':'同条件仅价格对照',
           'E3_FORWARD_EPS':'仅前瞻盈利','E4_EARNINGS_AND_PRICE_REGIME':'盈利与趋势同时确认','BUY_HOLD':'买入持有510300'}
    table=['| 方案 | 年化收益 | 夏普 | 最大回撤 |','|---|---:|---:|---:|']
    for x in result['all_metrics']:
        if x['cost']=='BASE':table.append(f"| {names[x['model']]} | {x['annualized_return']:.2%} | {x['net_sharpe']:.3f} | {x['max_drawdown']:.2%} |")
    text='''# 前瞻EPS策略：金融诊断结果与沪深300扩展

**前瞻盈利已从五份研报试验进入实际滚动训练和完整账户检验。金融诊断主方案夏普0.457，尚未达到1.2，也尚未证明稳定超额。历史沪深300主研究正在继续。**

本次交易研究标的仍只有510300和人民币现金，使用金融公司的盈利预期判断大盘ETF的持有比例，没有模拟买入金融个股。金融观察清单沿用十二家公司，其中十一家在固定机构目录中有原件；不代表全市场一致预期。

## 老板需要先知道的结果

核算期间为2020年1月2日至2026年8月14日开盘退出，初始20万元，计入整手、买卖费用、滑点、分红及应收。下面是基础费用下的完整账户结果，未截取有利时段。

'''+ '\n'.join(table)+'''

主方案累计收益21.97%，压力费用下夏普0.453、年化收益约3.00%。相对仅价格对照，增加前瞻盈利后点估计改善；但相对买入持有，主方案年化收益仍低约0.58个百分点。

主方案相对仅价格的年化算术收益增量约2.41个百分点，二十日区块估计的95%区间为负1.43至正6.52个百分点，包含零；主方案夏普区间约负0.191至1.206。区间上界超过1.2不表示策略达到1.2，实际完整账户点估计仍为0.457。

## 策略具体怎样做

每个月最后一个交易日收盘后形成判断，下一交易日开盘调整。先找出当时沪深300成分中有有效原始研报的观察公司，再提取以下三项盈利变量，与价格变量一起判断。

| 因子 | 中文计算规则 | 在策略中的作用 |
|---|---|---|
| 前瞻每股收益增长 | 在同一份研报中，用下一年度预测EPS减去当年度预测EPS，再乘二，除以两者绝对值之和；两者均为零则缺失 | 判断未来盈利是否改善，避免很小的盈利分母放大百分比 |
| 同年度预测利润修正 | 分别取判断日和九十日前当时有效报告，对比同一个绝对目标年度的预测净利润，按同样的对称变化计算 | 判断盈利预期是否上调，降低每股分母变化的干扰；净利润与归母净利润原文口径不一致时不配对 |
| 研报前瞻盈利收益率 | 取研报同表下一年度前瞻市盈率的倒数；市盈率为零或缺失则不使用 | 提供报告参考价格下的估值信息，不能称为判断日实时市盈率 |
| 六十日动量 | 用510300过去六十个交易日含分红收益形成累计价格方向 | 描述近期趋势 |
| 一百二十日均线距离 | 比较当前含分红财富水平与过去一百二十日平均水平 | 描述中期趋势位置 |
| 六十日波动率 | 用过去六十日收益波动估计年化风险 | 与预测收益共同决定研究持有比例 |

前三项在符合条件的公司中分别取中位数。金融诊断要求增长和估值至少五家公司、利润修正至少三家；历史成分主研究分别为三十家、十五家。缺失保留原状态，不把缺失当成增长为零。

研报必须在判断日前已经公开，以落款、发布、录入、目录日期中的较晚日期为信息日，且不在信息日当天使用。只看当时最新报告，超过180个日历日则过期；如果更新报告尚未提取出预测，原有观点也不能自动当成维持。跨年后仍锁定同一个预测年度计算修正。

每个月只形成一个训练样本。主方案将三项盈利变量和三项价格变量放入固定岭回归，预测未来六十个交易日的含分红收益；至少有十二个已经到达退出日的历史月份才拟合，正则强度固定为十。仅价格和仅前瞻盈利两个对照使用相同训练月份。

回归预测通过既有收益、风险和交易费用比较，在零、四分之一、二分之一、四分之三和全部五档权益比例中选择。风险厌恶系数固定为四，使用过去六十日方差，计入当前账户调仓费用。资料不足时不形成新观点，已有份额保持；第一次有效判断前保留初始现金。

另一个固定规则候选要求前瞻EPS增长为正、同年度利润预测修正为正、价格高于一百二十日均线，三项同时成立才持有，否则目标为现金；条件不完整则无新判断。该规则的夏普仅0.063，未显示简单叠加条件就能改善结果。

## 本次为什么仍未达标

**有效数据形成较晚。** 金融样本在2022年6月才首次形成符合固定条件的月度因子，共38个有效月末；累积够已成熟训练月份后，首次拟合在2024年6月28日。完整账户在2020至2023年仍保留初始现金，不能把此段从结果中删掉后重新称为全历史高夏普。

**控制风险尚未换来稳定超额。** 主方案完整期间平均股票暴露约19.5%，回撤低于买入持有，但年化收益也略低。金融诊断只证明这一组固定方法的点估计优于同条件价格对照，区间尚不能排除没有增量。

**单家机构、金融行业和少量有效月份仍限制判断。** 不能因金融诊断未达标就否定所有前瞻盈利研究，也不能因为某个阶段较好就推广。2024年至终点的主方案阶段夏普约0.728，同样低于1.2；已有历史还存在反复试验影响。

## 原件和因子已经修正到哪里

十二家公司的免费公开目录保存1,694份研报、37家机构。144个公司年度查询中，初次143个完成，唯一TLS连接失败已用同一地址正常重取，返回零条；原初结果和恢复记录分别保存。

固定国信目录选出154份金融研报，全部取得原件，其中139份提取417条年度EPS，另15份保留版式或口径缺口。报告日期存在差异的有58份，按较晚日期处理。年度列不能按接口当前年份替换；实际值、预测值、年度EPS和精确未来十二个月EPS也没有混用。

本轮保留未经股本调整的EPS九十日修正作诊断，但主方案不把送转、增发、回购等分母变化直接当成盈利上调。多数新版研报首页没有直接给出最新总股本，不能为了凑因子把未证明的股数填进去。下一阶段仍需补充股本和更多机构的连续来源。

## 主研究正在继续

固定国信机构的全行业目录取得7,726份研报，全部年度分页数一致。与2015年以来历史沪深300成分并集匹配后，选出3,352份原件，涉及369个历史成分证券；历史并集本身有668个证券代码，包含已经退出指数的公司。它不是每天的成分名单，因子汇总时仍按当天真实成分筛选。

2022年前公开目录记录很少，这是免费来源覆盖缺口，不证明当时没有分析师研究。当前正下载全部登记原件；原件阶段结束后，后续程序会依次执行EPS提取、月度特征和完整账户。历史成分与金融诊断的方法在本次金融收益读取前已一起冻结，不根据金融结果改参数。

公募净需求、全部期限逆回购和股东现金回报仍在总研究方向中。本轮为隔离前瞻盈利的增量作用，没有把尚待修复的全市场公募来源或未证明的股本变化直接塞入主模型。后续组合仍需新有限协议与实际完整账户检验。

## 本包范围

本包完整提供金融诊断的原始PDF和日期、417条预测、逐公司月度证据、全部训练记录与模型、十条完整账户及区间、代码、冻结配置和只读核对脚本。附带全行业和历史成分目录以及主研究冻结规则，正在下载的3,352份全行业原件不在本次金融诊断包中，不能称为主研究已经完成。

只读核对重新从139份原PDF读取对应页，核对417条EPS、139个月末、1518条公司月度记录、35个已保存成熟标签、75个拟合记录及十条账户；不重新训练，不重新抽随机样本。只进行必要的来源、数值和压缩包结构核对，没有外部GPT审阅或额外安全审计。
'''
    (DELIVERY/REPORT).write_text(text,encoding='utf-8')
    shutil.copy2(b/'510300_forward_eps_financial_facts_v1_1/年度前瞻每股收益_中文明细.csv',DELIVERY/'前瞻EPS原始预测_417条.csv')
    shutil.copy2(out/'每月前瞻盈利因子与覆盖.csv',DELIVERY/'月度因子与覆盖明细.csv')
    frame=pd.read_csv(out/'metrics.csv');frame['model']=frame.model.map(names)
    columns={'cost':'费用情景','model':'方案','annualized_return':'年化收益','net_sharpe':'夏普','max_drawdown':'最大回撤','trade_count':'交易次数','commission':'佣金','slippage_cost':'滑点成本'}
    frame[list(columns)].rename(columns=columns).to_csv(DELIVERY/'全部账户表现_中文对照.csv',index=False,encoding='utf-8-sig')
    (DELIVERY/'00_阅读入口.md').write_text('# 阅读入口\n\n先读《'+REPORT+'》，再看中文预测和账户明细。docs/510300_FORWARD_EPS_MONTHLY_POLICY_V1.md是完整中文规则。reports/research/510300_forward_eps_monthly_policy_v1_financial包含全部账本和训练记录。\n\n历史成分主研究仍在处理原件，本包提供冻结方法与目录，未提供尚未完成的主研究结果。01_GPT审阅提示.md可直接复制给外部审阅者。\n',encoding='utf-8')
    (DELIVERY/'01_GPT审阅提示.md').write_text('''# 请审阅前瞻盈利诊断并提出下一研究方向

目标是510300与人民币现金的完整账户成本后夏普至少1.2，并有稳定超额证据。用户要求主要研究前瞻EPS，免费来源，考虑股东回报、公募净需求及全部期限逆回购。此前十轮失败保留；本包是第十一轮金融诊断，主方案0.457未达标。历史沪深300主研究尚未完成。

请先核对原PDF的绝对预测年度、实际值与预测值、原报告日期、净利润口径和股数影响，再检查月度样本、成熟标签、同条件价格对照及完整账户费用。指出单家机构、金融样本、早期覆盖不足、资料过期、未解析新报告导致无观点和持仓延续会怎样影响结论。报告中的前瞻市盈率采用当时报告参考价格，尚非每个判断日更新的实时估值，这会造成哪些限制？

主方案比匹配价格对照的点估计好，但增量区间跨零，完整年化收益低于买入持有。请区分风险暴露、时点判断、条件样本和真实信息增量，不能仅凭夏普相对提高就宣布稳定超额。请具体评估正在按冻结方法扩展的历史成分研究，给出下一个有限候选、优先补充的免费来源、多机构与股本修正、完整账户对照、独立证据和停止条件。

不要把目录分页一致、保存数值核对或此包本身当成经济有效性、独立验证或交易许可。不要把区间上界1.206当成实际夏普。请用中文给出具体问题、影响、下一步骤和可验收标准。
''',encoding='utf-8')
    print('前瞻月度金融诊断中文说明已生成。',flush=True)


def package():
    if ZIP.exists():raise FileExistsError('本轮审阅包已经存在')
    if not (DELIVERY/REPORT).exists():raise FileNotFoundError('中文说明尚未准备')
    paths=set()
    for folder in FOLDERS:paths.update(p for p in (ROOT/'reports/research'/folder).rglob('*') if p.is_file())
    paths.update(ROOT/'config'/(name+'_manifest.json') for name in MANIFESTS)
    paths.update(ROOT/p for p in ['scripts/build_forward_eps_monthly_review_20260906.py','scripts/verify_forward_eps_monthly_saved_20260906.py',
        'scripts/continue_forward_eps_csi_pipeline_20260906.py','docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md',
        'config/510300_research_authority_v6.json','config/510300_adaptive_allocation_v1.json',
        'reports/research/510300_adaptive_allocation_v1/evaluation/BASE/BUY_HOLD_ledger.parquet',
        'reports/research/510300_adaptive_allocation_v1/evaluation/STRESS/BUY_HOLD_ledger.parquet',
        'reports/research/510300_forward_eps_csi_originals_v1/selected_before_originals.parquet',
        'reports/research/510300_forward_eps_csi_originals_v1/document_records/AP201811011229685729.json',
        'reports/research/510300_forward_eps_csi_originals_v1/page_texts/AP201811011229685729.json'])
    for name in ['RUN_STARTED.json','status.json']:
        p=ROOT/'reports/research/510300_forward_eps_csi_pipeline_v1'/name
        if p.exists():shutil.copy2(p,DELIVERY/('主研究接续程序现场快照_'+name))
    visited=set();frozen_refs=0
    def collect_raw(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ['raw_path','raw_pdf_path','raw_html_path','source_record','reused_from','source_raw_path'] and isinstance(v,str):
                    p=ROOT/v
                    if p.is_file():paths.add(p)
                elif isinstance(v,(dict,list)):collect_raw(v)
        elif isinstance(obj,list):
            for x in obj:collect_raw(x)
    while paths-visited:
        current=sorted(paths-visited)
        for p in current:
            visited.add(p)
            if not p.is_file():raise FileNotFoundError(p)
            if p.suffix=='.json':
                try:d=read(p)
                except (json.JSONDecodeError,UnicodeError):continue
                collect_raw(d)
                if p.name.endswith('_manifest.json'):
                    for x in d.get('files',[]):
                        q=ROOT/x['path']
                        if hashlib.sha256(q.read_bytes()).hexdigest()!=x['sha256']:raise ValueError('冻结引用变化：'+x['path'])
                        paths.add(q);frozen_refs+=1
            elif p.suffix=='.py':
                tree=ast.parse(p.read_text('utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        modules=[node.module]
                        if node.module in ['research','scripts']:modules.extend(node.module+'.'+a.name for a in node.names)
                        for module in modules:
                            if module.startswith(('research.','scripts.')):
                                q=ROOT/(module.replace('.','/')+'.py')
                                if q.is_file():paths.add(q)
    paths={p for p in paths if '__pycache__' not in p.parts}
    for p in sorted(paths):
        target=DELIVERY/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    requirements={n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','requests','pypdfium2','scikit-learn','joblib','pytest']}
    save(DELIVERY/'运行依赖版本.json',requirements,exclusive=True)
    (DELIVERY/'只读复核与范围.md').write_text('''# 只读复核与材料范围

安装运行依赖版本文件中的库后，从解压目录运行 scripts/verify_forward_eps_monthly_saved_20260906.py，即可从本包原件和账本复核金融诊断。该程序默认金融范围，重新读取原PDF对应页、核对保存的预测和月度证据、已保存标签、绩效和区间，不重新拟合模型、下载来源或生成账户。

本包提供本轮金融诊断全部直接数值输入、原件与完整账户，以及本轮各阶段冻结引用。复制的历史成分与ETF价格资料作为已冻结上游输入；其官方来源的完整上游重建过程、旧十轮全部策略账户，以及尚在下载的历史成分全部3352份原件不在此包范围。两条旧买入持有账本用于逐值比较基准一致性。

目录收集和账户研究的原执行脚本也在包中，用于查看方法；再次执行可能需要网络或会因已有完成结果拒绝覆盖。请优先运行只读核对脚本。本包没有独立外部GPT审阅，没有上传或额外安全审计，不构成交易授权。
''',encoding='utf-8')
    check=verify(DELIVERY,'financial');save(DELIVERY/'保存结果只读核对.json',check,exclusive=True)
    files=sorted(p for p in DELIVERY.rglob('*') if p.is_file() and p.name!='FILE_INDEX.csv')
    index=[{'path':p.relative_to(DELIVERY).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    with (DELIVERY/'FILE_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(index)
    building=ZIP.with_suffix('.building.zip')
    if building.exists():raise FileExistsError('临时压缩包已存在，需要检查现有结果')
    print('本包原件和十条账户只读核对通过，开始压缩和结构检查。',flush=True)
    with zipfile.ZipFile(building,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DELIVERY.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DELIVERY).as_posix())
    with zipfile.ZipFile(building) as z:
        names=z.namelist();assert len(names)==len(set(names));assert z.testzip() is None
        listed=list(csv.DictReader(io.StringIO(z.read('FILE_INDEX.csv').decode('utf-8-sig'))))
        assert set(names)=={x['path'] for x in listed}|{'FILE_INDEX.csv'}
        for x in listed:
            blob=z.read(x['path']);assert len(blob)==int(x['bytes']);assert hashlib.sha256(blob).hexdigest()==x['sha256']
    building.replace(ZIP)
    receipt={'completed_at':now(),'zip_path':ZIP.relative_to(ROOT).as_posix(),'bytes':ZIP.stat().st_size,
             'sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'members':len(names),'indexed_files':len(index),
             'frozen_file_references_checked':frozen_refs,'crc_and_duplicate_and_index_hash_checks':'PASS',
             'portable_saved_numerical_verification':check,'unfinished_csi_originals_excluded':True,
             'scope':'COMPLETE_FINANCIAL_FORWARD_EPS_DIAGNOSTIC_WITH_FROZEN_CSI_EXTENSION_DIRECTORY',
             'security_audit_performed':False,'external_review_performed':False}
    save(ZIP.with_suffix('.delivery.json'),receipt,exclusive=True);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--package',action='store_true');a=ap.parse_args()
    prepare() if a.prepare else package()
