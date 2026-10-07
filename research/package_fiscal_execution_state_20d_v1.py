"""建立本轮完整审阅包，索引核对及新解压目录只读数值复核。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_fiscal_execution_state_20d_v1'
REV=ROOT/'reports/research/510300_reversal_monthly_diagnostic_v1'
ARCHIVE=ROOT/'deliverables/510300_财政执行状态与均值回归检验_V1_GPT审阅_20260922.zip'
PARENT=ROOT/'deliverables/510300_官方政策目录与六类信息更新_V1_GPT审阅_20260922.zip'
PARENT_SHA='32d8156b700afd687435f105af9e808253096bc7c1e1301be7b2565e7e08f3d8'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare():
    if ARCHIVE.exists():raise FileExistsError('归档已完成，不能覆盖')
    if sha(PARENT)!=PARENT_SHA:raise ValueError('前阶段归档身份变化')
    (OUT/'history').mkdir(exist_ok=True);shutil.copy2(PARENT,OUT/'history'/PARENT.name)
    shutil.copytree(REV,OUT/'reversal_diagnostic',dirs_exist_ok=True)
    for name in ['verify_fiscal_execution_state_20d_v1.py','report_fiscal_execution_state_20d_v1.py','package_fiscal_execution_state_20d_v1.py','510300_reversal_monthly_diagnostic_v1.py']:
        shutil.copy2(ROOT/'research'/name,OUT/'code'/name)
    shutil.copy2(ROOT/'reports/research/510300_macro_dynamic_reframe_v1/inputs/money_104.csv',OUT/'inputs/money_for_chart_only.csv')
    shutil.copy2(ROOT/'reports/research/510300_macro_research_program_status_v1.json',OUT/'evidence/program_status_snapshot.json')
    shutil.copy2(ROOT/'reports/research/510300_policy_information_clock_v1/figures/510300_货币数据与六类政策_完整走势.png',OUT/'figures/前阶段_510300_M1M2剪刀差与六类政策.png')
    attachment=Path('E:/CodexData/.codex/attachments/9985585b-6e77-41bf-b927-0e02209a9c57/pasted-text-1.txt')
    shutil.copy2(attachment,OUT/'用户原始目标.txt')
    navigation='''# 阅读顺序

本包结论：财政执行及其与融资状态的固定交互未通过收益/风险增量门；510300固定二十日月末诊断也未证明反转明显多于延续。不是所有宏观研究或所有反转策略的否定。没有账户、卖点、订单或独立前向验证。

1. 研究结论.md：通俗结论、完整比较与边界。
2. figures：624日货币财政融资走势、收益风险对照，以及前阶段M1/M2剪刀差与政策图。
3. reversal_diagnostic/figures：170月末的反转与延续图；同目录保留全部CSV、输入和统计。
4. protocol.json、freeze.json：财政固定协议。两次无标签启动异常及修正记录单列，不篡改原冻结。
5. inputs、raw、evidence/acquisition：原始事实、当时可用上界和取得失败记录。127条正式财政数据的直接来源都已包含。
6. results：3476点日频、全部701个判断原点、497个逐期预测、88个主要评价、704个模型记录、所有对照、区块索引和逐年贡献。
7. code/verify_fiscal_execution_state_20d_v1.py：从新解压根目录运行，传入 --study-dir . --reversal-dir reversal_diagnostic；只检查保存证据。
8. GPT审阅提问.md：复制后与整个ZIP一起交给审阅者。

history保留上一阶段归档，事先验证SHA-256；它仍是原来的数据整理阶段，不升级为预测结论。FILE_INDEX.csv是本包成员索引，不等于科学或外部审核。
'''
    (OUT/'00_README_FIRST.md').write_text(navigation,encoding='utf-8')
    (OUT/'用户新增要求与研究边界.md').write_text('''用户要求同时研究货币、其他宏观政策、真实事前预期、持续更新和资金传导，并强调用真实图形表达；上涨较多后结合回撤退出；20万元主账户、2万元成本对照。最新补充是“A股均值回归、反转多于动量”。

本轮在财政模型冻结以前加入了方向受限的反转/动量对照，并另外固定月末原点、20日回报进行描述诊断。没有把个股横截面反转当成ETF择时既定事实。旧M1/M2五日、新订单及85/15终止状态保留。

财政子实验的失败只终止此固定表达；真实预期、政策冲击、资金持续流入与暂时压力回归、跨信息更新账户尚未全部完成。收益/风险门失败时不强行运行账户或卖点搜索，未运行指标保留NOT_RUN。
''',encoding='utf-8')
    (OUT/'GPT审阅提问.md').write_text('''请先读00_README_FIRST.md及研究结论.md，再复核来源、协议、全部比较和反转诊断。不要只挑最好的RMSE或图中的某段行情。

请重点检查：
1. 127个财政累计口径、2015年同口径、倍与百分比转换；119份HTTP原文和8份工具完整正文是否支持录入数值；未认证历史网页版本意味着什么。
2. 保守转载日期是否造成延迟或选择偏差，两个同日月份合并是否正确；不能把所属月份、网页创建日或后来公开日互相替换。
3. 训练标签成熟、同样可成交起点、入场日分红排除、DR007滞后、模型方向边界及共同起点更新是否成立。可运行只读verifier，不要重新调参。
4. 170月末中81反转88延续1零是否被准确解读；频数、回报自相关、均值平稳、横截面反转、盈利策略之间是否被混淆。
5. 财政收益轻微点值改善、风险未过门、动量斜率归零等结果是否支持当前有限结论。四候选局部校正没有覆盖上游全部历史选择。
6. 81个公告更新诊断中，少数方向变化和风险点值改善是否不足以证明连续持有价值。

除指出问题外，请给出后续独立策略方向的优先级、经济机制、必要数据与信息时钟、固定基准、验收和停止条件。优先说明怎样用政策信息和真实资金区分延续与回归；不要建议继续搜索本轮窗口、系数、方向或卖点以跨越历史目标线。宏观主线不因单一子实验失败而关闭；缺某项资金数据也不锁死其他独立通道。

整个历史已被观察，严格前向事件为0。包的结构及数值复核不是外部审阅，也没有证明年化10%或夏普1.2。无订单、Paper、Shadow、实盘或对外发布授权。
''',encoding='utf-8')
    (OUT/'交付范围与排除说明.md').write_text('''本包包含本轮研究事实、代码、来源、失败尝试、全部预测和反转诊断，不包含任何完整新账户，因为预测门未通过。不会用零收益占位。代码保留冻结原版、无标签修正版本及实际执行版本。

code/verify_fiscal_execution_state_20d_v1.py可以在独立解压目录使用Python、numpy、pandas、pyarrow、beautifulsoup4直接复核；研究执行脚本保留项目原路径入口和一次启动保护，并不承诺从任意目录重新拟合。提供全输入和源代码供审阅，不将一次结构核对声称为完全可重复研究或外部认证。

未重复打包DR007供应商全部原始响应及其历史首次传输凭证；继承parquet带来源身份、原始响应集合哈希和使用规则。未重新下载行情全历史原始供应商包；保存研究使用的完整日频表和含分红恒等式。M1/M2及六类政策背景来源通过上一阶段自包含ZIP保留。

来源字段公开日为保守日期上界，不保证市场首发时间；8份网页工具完整正文明确标为非HTTP原文。所有财政预期为空。没有把缺失资料说成已补齐市场全部信息。

不包含凭证、账户、与研究无关的工作区修改、虚拟环境、缓存或其他大规模研究族；当前包本身和外部交付回执不自包含，以避免自指哈希。
''',encoding='utf-8')
    save(OUT/'evidence/图表目视核对.json',dict(checked_at=now(),status='PASS_VISUAL_INSPECTION',figures=['figures/510300_货币财政融资_走势对比.png','figures/财政执行_收益风险与均值回归对照.png','reversal_diagnostic/figures/510300_二十日反转与延续对比.png'],daily_points=624,reversal_origins=170,corrections=['将图例移出数据区，修复区间标注边界和高低点位置','固定风险比较行的显式顺序，避免R2相对R1与R2相对R0标签互换'],prediction_data_changed=False))
    save(OUT/'evidence/前阶段归档身份.json',dict(archive=PARENT.name,sha256=PARENT_SHA,status='HASH_VERIFIED_PRESERVED_PRIOR_STAGE'))
    print('导航、用户需求、源码、来源、全部结果及前阶段归档已整理。')


def package():
    if ARCHIVE.exists():raise FileExistsError('最终ZIP已存在，不能覆盖')
    command=[sys.executable,str(OUT/'code/verify_fiscal_execution_state_20d_v1.py'),'--study-dir',str(OUT),'--reversal-dir',str(OUT/'reversal_diagnostic')]
    run=subprocess.run(command,check=True,capture_output=True,text=True,encoding='utf-8');checked=json.loads(run.stdout)
    save(OUT/'evidence/保存结果独立复核.json',checked)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['FILE_INDEX.csv','delivery_receipt.json'] and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
    entries=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(entries);index=buffer.getvalue().encode('utf-8-sig')
    building=ARCHIVE.with_suffix('.building.zip')
    with zipfile.ZipFile(building,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files:z.write(p,p.relative_to(OUT).as_posix())
        z.writestr('FILE_INDEX.csv',index)
    if building.stat().st_size>80*1024*1024:raise ValueError('超过本轮80MiB包范围')
    with tempfile.TemporaryDirectory(prefix='fiscal_review_verify_') as temp:
        with zipfile.ZipFile(building) as z:
            names=z.namelist()
            if z.testzip() is not None or len(names)!=len(set(names)):raise ValueError('ZIP CRC或重复成员失败')
            if set(names)!={r['path'] for r in entries}|{'FILE_INDEX.csv'}:raise ValueError('成员索引不一致')
            for row in entries:
                data=z.read(row['path'])
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('索引大小/哈希不一致')
            z.extractall(temp)
        replay=subprocess.run([sys.executable,str(Path(temp)/'code/verify_fiscal_execution_state_20d_v1.py'),'--study-dir',temp,'--reversal-dir',str(Path(temp)/'reversal_diagnostic')],check=True,capture_output=True,text=True,encoding='utf-8')
        if json.loads(replay.stdout)!=checked:raise ValueError('新解压目录只读复算不一致')
    building.replace(ARCHIVE);(OUT/'FILE_INDEX.csv').write_bytes(index)
    receipt=dict(created_at=now(),archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),members=len(entries)+1,indexed_members=len(entries),crc='PASS',duplicates=0,index_size_hash='PASS',fresh_extraction_recomputation=checked,status='PASS_STRUCTURAL_AND_SAVED_RECOMPUTATION',external_review_completed=False,whole_macro_objective_complete=False,new_accounts=0)
    save(ARCHIVE.with_suffix('.receipt.json'),receipt);save(OUT/'delivery_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='财政与反转研究自包含审阅包');parser.add_argument('action',choices=['prepare','package']);args=parser.parse_args();{'prepare':prepare,'package':package}[args.action]()
