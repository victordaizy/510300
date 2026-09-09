"""交付前瞻EPS两轮完整结果：数值审阅包及含全部直接PDF的完整包。"""
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
from scripts.verify_forward_eps_monthly_saved_v1_1_20260906 import verify as verify_monthly
from scripts.verify_forward_eps_attribution_saved_20260906 import verify as verify_attribution
from scripts.verify_forward_eps_residual_saved_20260906 import verify as verify_residual

DELIVERY=ROOT/'deliverables/510300前瞻EPS沪深300与完整价格基线_20260906'
CORE=ROOT/'deliverables/510300前瞻EPS_沪深300与完整价格基线_GPT数值审阅_20260906.zip'
FULL=ROOT/'deliverables/510300前瞻EPS_沪深300与完整价格基线_完整原件_20260906.zip'
REPORT='前瞻EPS策略_沪深300结果与下一步.md'
NAMES={'E1_PRICE_FORWARD_EPS':'价格加前瞻盈利','E2_MATCHED_PRICE':'同覆盖价格对照','E3_FORWARD_EPS':'仅前瞻盈利',
       'E4_EARNINGS_AND_PRICE_REGIME':'盈利与趋势同时确认','BUY_HOLD':'买入持有510300',
       'S1_PRICE_PLUS_EPS_RESIDUAL':'完整价格基线加EPS误差修正','S2_PRICE_ALL_MATURE_MONTHS':'全部成熟月份价格基线'}


def identity(path):
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def table(result):
    rows=['| 方案 | 费用情景 | 年化收益 | 夏普 | 最大回撤 |','|---|---|---:|---:|---:|']
    for row in result['all_metrics']:
        sharpe='无法计算' if row['net_sharpe'] is None else f"{row['net_sharpe']:.3f}"
        rows.append(f"| {NAMES[row['model']]} | {'基础' if row['cost']=='BASE' else '压力'} | {row['annualized_return']:.2%} | {sharpe} | {row['max_drawdown']:.2%} |")
    return '\n'.join(rows)


def prepare():
    if (DELIVERY/REPORT).exists():raise FileExistsError('本次中文交付已经准备，不覆盖')
    base=ROOT/'reports/research';csi=read(base/'510300_forward_eps_monthly_policy_v1_csi/result.json')
    financial=read(base/'510300_forward_eps_monthly_policy_v1_financial/result.json')
    residual=read(base/'510300_forward_eps_residual_policy_v1/result.json')
    facts=read(base/'510300_forward_eps_csi_facts_v1_1/result.json')
    source=read(base/'510300_forward_eps_csi_originals_v1/result.json')
    coverage=read(base/'510300_forward_eps_monthly_policy_v1_csi/source_receipt.json')
    main=csi['primary'][0];second=residual['primary'][0]
    DELIVERY.mkdir(parents=True,exist_ok=True)
    text=f'''# 前瞻EPS策略：沪深300结果与完整价格基线

**前瞻EPS已完成历史沪深300主研究，并继续完成了“完整价格基线加盈利误差修正”试验。第十一轮主方案基础费用夏普{main['net_sharpe']:.3f}，第十二轮主方案{second['net_sharpe']:.3f}。研究目标仍为完整成本后夏普至少1.2并有稳定超额证据；本包不把历史点估计自动视为目标完成。**

所有账户仅包含510300和人民币现金，初始二十万元，评价从2020年1月2日至2026年8月14日开盘退出。包含整手、佣金最低五元、滑点、分红权益及应收、T+1和涨跌停约束。基础佣金万分之二、单边滑点万分之五；压力佣金万分之四、单边滑点千分之一。现金收益和夏普参考无风险收益都按此前登记的零计算。

## 第十一轮：历史沪深300成分主研究

{table(csi)}

本轮有{coverage['valid_monthly_origins']}个满足固定EPS覆盖条件的月末，首次为{coverage['first_valid_month']}。原件来自同一家国信证券，不能称为全市场一致预期。每天按历史真实成分筛选，没有把如今的成分名单倒推至过去。

## 第十二轮：完整价格基线与前瞻EPS误差修正

{table(residual)}

新增方案先让价格模型使用全部已成熟的有效价格月份；再让EPS模型解释该价格模型过去预测的误差。训练误差只采用历史当时保存的价格预测，标签须在本次判断之前已经兑现。EPS资料缺失或误差样本不足时，明确使用独立价格基线，同时保留EPS缺失状态。

保存了{residual['trained_price_models']}个价格模型、{residual['trained_eps_residual_models']}个盈利修正模型；评价期内实际有{residual['evaluation_eps_adjustment_months']}个月末加入了EPS修正。两候选加买入持有，两档费用共六条完整账户。二十日、六十日区块各两千次共同抽样的区间均保存，不能只选更有利的一组。

## 每个因子的中文规则

| 因子 | 具体计算与含义 |
|---|---|
| 前瞻EPS增长 | 同份报告下一年度预测EPS减去当年度预测EPS，乘二后除以两者绝对值之和；均为零则缺失。比较同一份报告可减少股本口径不同的影响。 |
| 同年度预测利润修正 | 在判断日和九十日前各取当时有效报告，始终比较同一绝对目标年度的预测利润，按上述对称变化计算。净利润与归母净利润原文口径不同时不配对。 |
| 研报前瞻盈利收益率 | 取原报告同表下一年度市盈率的倒数；零或缺失不可用。它对应报告参考价格，不等于判断日实时市盈率。 |
| 六十日动量 | 用510300过去六十交易日的含分红财富变化表示近期方向。 |
| 一百二十日均线距离 | 比较当前含分红财富水平和过去一百二十日均值，表示中期趋势位置。 |
| 六十日波动率 | 从过去六十交易日收益估计风险，参与预测和持仓选择。 |

三项盈利变量分别对当时符合条件的公司取中位数。沪深300主研究要求增长和估值各至少三十家公司、预测利润修正至少十五家。预测年度采用原报告明确年度，不用接口当前年份替换；年度EPS不等于精确未来十二个月EPS。

每月最后交易日收盘判断、下一交易日开盘调仓；只采用信息日严格早于判断日的报告。信息日取报告落款、发布、录入、目录日期中的较晚者；超过一百八十个日历日过期。新报告尚不能提取预测时，旧观点不能自动视作维持。每月只算一个训练样本，防止把同一份报告日复一日当成大量独立数据。

第十一轮主模型是三项价格加三项盈利变量的岭回归，固定正则强度十、至少十二个已兑现月份；仅价格和仅EPS是同覆盖对照。规则候选则要求EPS增长、利润修正和一百二十日趋势同时为正。

第十二轮完整价格基线至少使用二十四个成熟月份，带截距岭回归；EPS修正至少十二个成熟误差月份，用标准化盈利变量、强度十且不另加截距的岭回归。所有预测都通过相同的收益、风险和费用比较，在零、四分之一、二分之一、四分之三、全部五档权益比例中选择；风险厌恶系数四、既有调仓带宽保持一致。

## 金融诊断与失败原因

{table(financial)}

金融主方案完整夏普0.457，2024年6月首次预测，2025年1月首次实际买入。初次训练只有十三个月份，集中在2022至2023年。新增分解发现：从模型可用后的共同期间看，相对价格对照的收益改善主要对应更多持仓，持仓变化关联项略为负且区间跨零。详见本包“金融诊断_失败原因与持仓分解.md”。

本包同时提供沪深300账户的相同分解，不将事后平均暴露或变化关联直接解释为因果择时能力。相关表内年化算术收益增量，与复利年化收益是不同统计量，分别标注。

## 已有资料和实际限制

原件队列固定为3,352份，覆盖369个历史成分证券；全部目录、成功及缺口记录都保存。实际原件汇总和预测事实计数见“原件与事实完成记录.json”。首次取得的历史报告不是当年实时归档的不可变快照，这项限制保留。

本轮主要检验前瞻盈利信息。股东回报、公募净申购及全部期限逆回购继续列入后续研究；ETF账户实际分红已经记账。股票回购对股数和EPS的影响、公募全市场净需求、判断日实时前瞻估值尚需各自一致的时点与口径，不能直接用不完整代理填入。

此前十轮结果、本轮金融结果及过去反复读取的历史都不构成新的独立样本。完整账户夏普、相对买入持有收益、阶段稳定性和不确定性必须一起看；区间上界达到1.2不能当作目标达成。

## 文件范围

“GPT数值审阅”包保留全部直接数值输入、原报告提取的逐页文本、日期和来源记录、预测事实、模型、完整账户、抽样文件、冻结规则与复核脚本，供单个文件上传。因为原始PDF合计超过3GB，该包外置原始PDF，详见“外置原始PDF索引.csv”。它能独立核对已保存原文和账户数值，不能在没有PDF字节的条件下重新证明原始版式。

“完整原件”包补齐上述全部直接PDF，使用相同相对路径，保留完整数据来源。两包均保留公开原链接与全文指纹。本地还对原始PDF重新提取对应页，核对事实；这项检查与数值包内的原文核对分别记录。旧十轮全部原件及上游价格和历史成分的完整重建过程不在本次包中；相关资料作为已冻结上游输入。

只进行了必要的来源、数值和压缩结构检查，没有外部GPT审阅或额外安全审计，没有交易授权。
'''
    (DELIVERY/REPORT).write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'docs/510300_FORWARD_EPS_FINANCIAL_FAILURE_ATTRIBUTION_20260906.md',DELIVERY/'金融诊断_失败原因与持仓分解.md')
    save(DELIVERY/'原件与事实完成记录.json',{'originals':source,'facts':facts,'source_coverage':coverage},exclusive=True)
    all_rows=[]
    for label,result in [('金融诊断',financial),('沪深300主研究',csi),('完整价格与EPS误差修正',residual)]:
        for row in result['all_metrics']:
            all_rows.append({'研究':label,'方案':NAMES[row['model']],'费用情景':'基础' if row['cost']=='BASE' else '压力',
                             '年化收益':row['annualized_return'],'夏普':row['net_sharpe'],'最大回撤':row['max_drawdown'],
                             '累计收益':row['cumulative_return'],'佣金':row['commission'],'滑点':row['slippage_cost']})
    pd.DataFrame(all_rows).to_csv(DELIVERY/'全部26条完整账户_中文对照.csv',index=False,encoding='utf-8-sig')
    (DELIVERY/'00_阅读入口.md').write_text('# 阅读入口\n\n先读《'+REPORT+'》，再看“全部26条完整账户_中文对照.csv”和金融失败分解。完整中文规则在docs目录。reports/research包含原件逐页文本、绝对年度预测、全部月份、模型、账本和抽样结果。\n\n数值审阅包外置全部原始PDF，范围与索引明确；完整原件包补齐这些PDF。请复制“01_GPT审阅提示.md”给外部审阅者，先检查证据，再要求提出下一组有限研究方案。\n',encoding='utf-8')
    (DELIVERY/'01_GPT审阅提示.md').write_text('''# 请审阅前瞻EPS研究，并提出下一项可执行研究

目标是510300与人民币现金，完整成本后夏普至少1.2并有稳定超额证据。主要研究前瞻EPS，全部来源免费。请分别检查第十一轮同覆盖对照和第十二轮完整价格基线加盈利误差修正，不要把金融子样本代表整个指数。

请核对绝对年度EPS、同年度预测利润修正、原报告参考估值、股本分母、日期取较晚者、历史真实成分、最新未解析报告导致缺失、覆盖阈值和历史资料可用性。说明目前来源仍是单一机构，年度EPS不是精确未来十二个月，哪些缺口最影响结论。

检查价格基线是否使用全部已兑现月份，EPS误差标签是否来自该月当时保存的预测，是否混入未来训练或重写旧预测。检查完整账户佣金最低额、滑点、T+1、分红应收和终点退出。比较全部期间、分年、三阶段、两档费用和二十日及六十日区间，不得挑选区间或时代宣布达标。

请结合持仓规模和持仓变化分解，区分更高平均暴露、资料晚到、阶段集中与潜在信息增量。事后分解不是可交易策略或因果证明。公募净需求、全部期限逆回购与股东回报应怎样和前瞻EPS形成有限、可证伪的下一步，而不是不断堆叠参数？

给出具体缺陷、影响、优先免费来源、下一组少量候选、训练与独立验证安排，以及停止条件。数值审阅包的PDF外置列表是明确范围限制；若需要重新审阅原始版式，请索取同名完整原件包。结构和数值核对不等于经济有效、外部独立审阅或交易许可。
''',encoding='utf-8')
    print('沪深300及完整价格基线的中文报告已准备。',flush=True)


def gather():
    base=ROOT/'reports/research'
    folders=['510300_forward_eps_history_directory_v1','510300_forward_eps_csi_directory_v1','510300_forward_eps_csi_pipeline_v1',
             '510300_forward_eps_financial_originals_v1','510300_forward_eps_csi_originals_v1',
             '510300_forward_eps_financial_facts_v1','510300_forward_eps_financial_facts_v1_1','510300_forward_eps_csi_facts_v1_1',
             '510300_forward_eps_monthly_policy_v1_financial','510300_forward_eps_monthly_policy_v1_csi',
             '510300_forward_eps_exposure_attribution_v1_financial','510300_forward_eps_exposure_attribution_v1_1_financial',
             '510300_forward_eps_exposure_attribution_v1_2_financial','510300_forward_eps_exposure_attribution_v1_2_csi',
             '510300_forward_eps_residual_policy_v1']
    paths={p for folder in folders for p in (base/folder).rglob('*') if p.is_file()}
    manifests=['history_directory_v1','csi_directory_v1','financial_originals_v1','csi_originals_v1','financial_facts_v1','financial_facts_v1_1',
               'csi_facts_v1','csi_facts_v1_1','monthly_policy_v1','exposure_attribution_v1','exposure_attribution_v1_1','exposure_attribution_v1_2','residual_policy_v1']
    paths.update(ROOT/f'config/510300_forward_eps_{name}_manifest.json' for name in manifests)
    paths.update(ROOT/p for p in ['scripts/build_forward_eps_csi_residual_review_20260906.py','scripts/verify_forward_eps_monthly_saved_v1_1_20260906.py',
        'scripts/verify_forward_eps_attribution_saved_20260906.py','scripts/verify_forward_eps_residual_saved_20260906.py',
        'scripts/continue_forward_eps_csi_pipeline_20260906.py','docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md',
        'docs/510300_FORWARD_EPS_FINANCIAL_FAILURE_ATTRIBUTION_20260906.md','config/510300_research_authority_v6.json','config/510300_adaptive_allocation_v1.json'])
    visited=set();frozen=[]
    def raw_refs(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ['raw_path','raw_pdf_path','raw_html_path','source_record','reused_from','source_raw_path'] and isinstance(value,str):
                    path=ROOT/value
                    if path.is_file():paths.add(path)
                elif isinstance(value,(dict,list)):raw_refs(value)
        elif isinstance(obj,list):
            for value in obj:raw_refs(value)
    while paths-visited:
        for path in sorted(paths-visited):
            visited.add(path)
            if not path.is_file():raise FileNotFoundError(path)
            if path.suffix=='.json':
                try:obj=read(path)
                except (json.JSONDecodeError,UnicodeError):continue
                raw_refs(obj)
                if path.name.endswith('_manifest.json'):
                    for item in obj.get('files',[]):
                        target=ROOT/item['path'];actual=identity(target)
                        assert actual['sha256']==item['sha256'];paths.add(target);frozen.append(item)
            elif path.suffix=='.py':
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        modules=[node.module]
                        if node.module in ['scripts','research']:modules.extend(node.module+'.'+n.name for n in node.names)
                        for module in modules:
                            if module.startswith(('research.','scripts.')):
                                target=ROOT/(module.replace('.','/')+'.py')
                                if target.is_file():paths.add(target)
    return {p for p in paths if '__pycache__' not in p.parts},frozen


def index_bytes(items):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(items)
    return stream.getvalue().encode('utf-8-sig')


def archive(destination,mapping,scope,max_bytes=None):
    if destination.exists():raise FileExistsError('目标压缩包已存在，不覆盖')
    indexed=[]
    for name,path in sorted(mapping.items()):
        indexed.append({'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    raw_index=index_bytes(indexed);temporary=destination.with_suffix('.building.zip')
    print('开始压缩',scope,len(indexed),'个材料文件',flush=True)
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,path in sorted(mapping.items()):z.write(path,name)
        z.writestr('FILE_INDEX.csv',raw_index)
    if max_bytes is not None and temporary.stat().st_size>max_bytes:raise ValueError('数值审阅包超过预留容量，保留临时包供处理')
    print('开始核对压缩结构和文件索引',scope,flush=True)
    with zipfile.ZipFile(temporary) as z:
        members=z.namelist();assert len(members)==len(set(members));assert z.testzip() is None
        entries=list(csv.DictReader(io.StringIO(z.read('FILE_INDEX.csv').decode('utf-8-sig'))))
        assert set(members)=={row['path'] for row in entries}|{'FILE_INDEX.csv'}
        for row in entries:
            value=z.read(row['path']);assert len(value)==int(row['bytes']);assert hashlib.sha256(value).hexdigest()==row['sha256']
    temporary.replace(destination)
    receipt={'completed_at':now(),'zip_path':destination.relative_to(ROOT).as_posix(),'bytes':destination.stat().st_size,
             'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'members':len(members),'indexed_files':len(indexed),
             'scope':scope,'crc_duplicate_index_size_hash_checks':'PASS','security_audit_performed':False,'external_review_performed':False}
    save(destination.with_suffix('.delivery.json'),receipt,exclusive=True);print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


def package():
    if not (DELIVERY/REPORT).exists():raise FileNotFoundError('中文报告尚未准备')
    if CORE.exists() or FULL.exists():raise FileExistsError('已有交付结果，应先核对进程和现有包，不重复打包')
    paths,frozen=gather();pdfs={p for p in paths if p.suffix.lower()=='.pdf'}
    for path in sorted(paths-pdfs):
        target=DELIVERY/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    pdf_index=[identity(p) for p in sorted(pdfs)]
    (DELIVERY/'外置原始PDF索引.csv').write_bytes(index_bytes(pdf_index))
    save(DELIVERY/'材料范围.json',{'registered_at':now(),'full_original_pdf_files':len(pdfs),'full_original_pdf_bytes':sum(x['bytes'] for x in pdf_index),
        'numeric_review_maximum_bytes':480000000,'numeric_review_excludes_pdf_bytes':True,'full_archive_includes_all_listed_pdfs':True,
        'frozen_references_checked':len(frozen),'no_old_zip_expanded':True},exclusive=True)
    save(DELIVERY/'运行依赖版本.json',{name:importlib.metadata.version(name) for name in ['numpy','pandas','pyarrow','requests','pypdfium2','scikit-learn','joblib','pytest']},exclusive=True)
    (DELIVERY/'只读复核说明.md').write_text('''# 只读复核

使用 scripts/verify_forward_eps_monthly_saved_v1_1_20260906.py，范围选择 financial 或 csi，数值审阅包需明确选择 saved_page_text 模式，核对保存的原文、预测、标签、模型及账户。完整原件包可用默认 original_pdf 模式，重新提取原PDF前三页并比对保存原文。

scripts/verify_forward_eps_attribution_saved_20260906.py 核对两个范围的保存分解与区块抽样文件。scripts/verify_forward_eps_residual_saved_20260906.py 核对第十二轮当时预测、成熟误差、模型参数与六条完整账户。上述程序不重新训练、生成账户、访问网络或抽取随机样本。

请优先使用只读程序；原采集、模型及账户程序用于理解规则，再次运行会受已有结果保护。源码中的字段名只是实现；策略规则已在中文报告与docs中完整说明。
''',encoding='utf-8')
    checks={'financial':verify_monthly(DELIVERY,'financial','saved_page_text'),'csi':verify_monthly(DELIVERY,'csi','saved_page_text'),
            'financial_attribution':verify_attribution(DELIVERY,'financial'),'csi_attribution':verify_attribution(DELIVERY,'csi'),
            'residual_policy':verify_residual(DELIVERY)}
    save(DELIVERY/'数值包保存结果独立核对.json',checks,exclusive=True)
    mapping={p.relative_to(DELIVERY).as_posix():p for p in DELIVERY.rglob('*') if p.is_file() and p.name!='FILE_INDEX.csv'}
    core=archive(CORE,mapping,'ALL_SAVED_TEXT_FACTS_MODELS_26_ACCOUNTS_AND_ATTRIBUTION_PDF_BYTES_EXCLUDED',480000000)
    mapping.update({p.relative_to(ROOT).as_posix():p for p in pdfs})
    full=archive(FULL,mapping,'ALL_DIRECT_PDFS_PLUS_COMPLETE_NUMERICAL_REVIEW_MATERIALS')
    save(ROOT/'reports/research/510300_forward_eps_csi_residual_delivery_20260906.json',{'completed_at':now(),'core':core,'full':full,
        'portable_saved_numerical_checks':checks,'frozen_references_checked':len(frozen),'pdf_files':len(pdfs)},exclusive=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--package',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else package()
