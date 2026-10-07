"""保存来源核对、生成2019年指数信息时间线及说明。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_index_information_sequence_v27'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sources_and_events():
    # 网络工具已读到正文，但直接下载未成功；只保存提取事实，明确不是原始HTML。
    web_notes = [
        {'id': 'rrr_pbc_reprint',
         'url': 'https://dfjrjgj.hunan.gov.cn/dfjrjgj/tslm_71665/mtjj/201905/t20190507_5328854.html',
         'title': '中国人民银行有关负责人表示：建立对中小银行实行较低存款准备金率的政策框架 深化金融供给侧结构性改革',
         'site_published_at': '2019-05-07T15:05:00+08:00',
         'publisher': '湖南省地方金融管理局转载中国人民银行问答',
         'facts': {'qualifying_banks_approx': 1000, 'announced_funding_yi_approx': 2800,
                   'qualifying_rrr_percent': 8, 'phases': ['2019-05-15', '2019-06-17', '2019-07-15'],
                   'purpose': '资金用于民营和小微企业贷款',
                   'qualification': '仅在本县经营，或跨县设分支但资产规模小于100亿元的农村商业银行'},
         'clock_limit': '这是5月7日转载时间，不能据此认定5月6日开盘前已经公开。',
         'retrieval_form': 'web工具读取正文后的结构化摘记；直接下载TLS失败；未保存原始HTML'},
        {'id': 'china_tariff',
         'url': 'https://gss.mof.gov.cn/gzdt/zhengcefabu/201905/t20190513_3256787.htm',
         'title': '国务院关税税则委员会发布公告决定对原产于美国的部分进口商品提高加征关税税率',
         'site_publication_date': '2019-05-13',
         'publisher': '财政部网站 国务院关税税则委员会办公室',
         'facts': {'effective_at': '2019-06-01T00:00:00+08:00', 'scope': '已加征关税的600亿美元美国商品中的部分',
                   'new_rates_percent': [25, 20, 10], 'remaining_rate_percent': 5},
         'clock_limit': '未核实小时分钟；用北京时间日期日终作公开时间保守上界。',
         'retrieval_form': 'web工具读取正文后的结构化摘记；直接下载HTTP 502；未保存原始HTML'},
    ]
    for note in web_notes:
        note['read_at'] = now()
        note['historical_immutable_vintage_proven'] = False
    save('sources/网页读取结构化摘记.json', web_notes)
    metadata = json.loads((OUT / 'source_receipts.json').read_text(encoding='utf-8'))
    keywords = {
        'money_march': ['2019-04-12 16:00:03', '8.6%', '4.6%'],
        'money_april': ['2019-05-09 10:00:03', '8.5%', '2.9%', '5347'],
        'pmi_april': ['2019/04/30 09:00', '50.1%', '51.4%'],
        'economy_q1': ['2019/04/17 10:00', '6.4%', '8.5%'],
        'politburo': ['2019-04-19 17:23:19', '结构性去杠杆', '松紧适度'],
        'tariff_intent': ['May 05, 2019 16:08:46', '25% on Friday'],
        'tariff_notice': ['May 9, 2019', '2019–09681', 'May 10, 2019'],
        'ustr_may10': ['May 10, 2019', '10 percent to 25 percent'],
        'economy_april': ['2019/05/15 10:00', '5.4%', '7.2%'],
    }
    reviewed = []
    for s in metadata:
        row = dict(s)
        if s['id'] in keywords:
            txt = (OUT / 'sources' / (s['id'] + '.txt')).read_text(encoding='utf-8')
            compact = ''.join(txt.split())
            checks = {v: ''.join(v.split()) in compact for v in keywords[s['id']]}
            if not all(checks.values()):
                raise AssertionError((s['id'], checks))
            row.update(content_review='核对正文日期与关键事实', keyword_checks=checks)
        elif s['id'] == 'china_tariff':
            row.update(content_review='网络工具已读正文；提取事实另存，无原始HTML')
        else:
            row.update(content_review='政府网原链接未取得；央行问答政府转载另列')
        reviewed.append(row)
    save('source_content_review.json', reviewed)
    events = [
        ('money_march', '2019-04-12T16:00:03+08:00', '网页标注到秒', '3月货币及一季度信贷',
         'M1 4.6%、M2 8.6%，剪刀差−4.0；一季度企业贷款多增以票据和短贷为主。',
         '经济数量和结构更新；不能直接换成沪深300盈利增速。', ''),
        ('economy_q1', '2019-04-17T10:00:00+08:00', '网页标注到分', '一季度GDP与3月生产',
         '首次公布一季度GDP同比6.4%；3月规上工业增加值同比8.5%。',
         '本轮未保存事前一致预期，不能把好数据直接命名为超预期冲击。', ''),
        ('politburo', '2019-04-19T17:23:19+08:00', '新华社标注到秒', '政治局经济工作表述',
         '讨论经济下行压力，强调结构性去杠杆、财政加力提效与稳健货币松紧适度。',
         '可能影响政策预期；原文没有宣布加息或全面货币收紧。', ''),
        ('pmi_april', '2019-04-30T09:00:00+08:00', '网页标注到分', '4月PMI与订单',
         '制造业PMI 50.1，较前月低0.4；新订单51.4，低0.2。',
         '需求扩张边际放缓，仍高于50；PMI覆盖不是沪深300盈利覆盖。', ''),
        ('tariff_intent', None, '档案时分秒未注明时区', '5月5日美国总统关税表态',
         '公开表达将2000亿美元清单商品的10%税率提高至25%的意图；意图与法律生效分开。',
         '出口与风险溢价的候选新信息；不能把整个假期跳空归因于单条消息。', ''),
        ('rrr_pbc_reprint', '2019-05-07T15:05:00+08:00', '政府转载时间，不是原始宣布时间', '中小银行降准政策的央行问答',
         '约1000家县域农商行适用8%准备金率，约2800亿元用于民营小微信贷；三期落地。',
         '结构性银行资金供给，不等同于510300成分股立即获得现金或盈利。', '2019-05-15/2019-06-17/2019-07-15'),
        ('money_april', '2019-05-09T10:00:03+08:00', '央行网页标注到秒；盘中重建', '4月货币信贷更新',
         'M1 2.9%、M2 8.5%，剪刀差−5.6；3个月企业中长期贷款由同比多增转为少增。',
         '原窗口内信息已经更新；不能继续将所有日子标签固定为3月货币改善。', ''),
        ('tariff_notice', '2019-05-10T11:59:59+08:00', '5月9日美国东部日终的保守上界', '美国关税法律通知',
         '5月9日联邦公报载明2000亿美元清单10%升至25%；适用于规定的入境与出口条件。',
         '此前已有5月5日政策意图；不能把本次文件日期当作最早消息日期。', '2019-05-10T12:01:00+08:00'),
        ('ustr_may10', '2019-05-11T11:59:59+08:00', '5月10日美国东部日终的保守上界', '美方实施及后续程序声明',
         '确认部分关税提高，并启动对其余约3000亿美元商品提高关税的程序。',
         '启动程序不等于其余商品当日已经全部加税。', ''),
        ('china_tariff', '2019-05-13T23:59:59+08:00', '只有发布日期，取北京时间日终', '中方回应关税公告',
         '部分600亿美元美国商品税率计划提高至25%、20%或10%，另有5%税目保持。',
         '窗口内是已宣布的未来政策，实际生效在原窗口之后。', '2019-06-01T00:00:00+08:00'),
        ('economy_april', '2019-05-15T10:00:00+08:00', '网页标注到分', '4月经济数据',
         '工业增加值同比5.4%；社零7.2%。同稿指出节假日因素调整后社零测算8.7%。',
         '首发数据偏弱与季节因素并存；不能把原始增速下降全部解释为需求恶化。', ''),
    ]
    urls = {s['id']: s['url'] for s in metadata}
    urls.update({s['id']: s['url'] for s in web_notes})
    market = pd.read_csv(OUT / 'inputs/market.csv', parse_dates=['date'])
    opens = market.date.dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=9, minutes=30)
    rows = []
    for identifier, at, precision, name, facts, limitation, effective in events:
        first_open = opens.loc[opens >= pd.Timestamp(at)].iloc[0].isoformat() if at else None
        rows.append({'event_id': identifier, 'public_clock_beijing_used': at, 'clock_precision_and_limit': precision,
                     'event': name, 'facts': facts, 'interpretation_limit': limitation, 'effective_at': effective,
                     'first_open_after_used_clock': first_open, 'source_url': urls[identifier],
                     'known_at_original_E0_open_by_this_clock': bool(at and pd.Timestamp(at) < pd.Timestamp('2019-04-15T09:30:00+08:00')),
                     'historical_immutable_vintage_proven': False})
    pd.DataFrame(rows).to_csv(OUT / 'results/信息时间线_事实与解释边界.csv', index=False, encoding='utf-8-sig')
    zones = []
    for zone in ['UTC', 'America/New_York', 'America/Los_Angeles']:
        moment = pd.Timestamp('2019-05-05 16:08:46', tz=zone).tz_convert('Asia/Shanghai')
        zones.append({'assumed_archive_zone': zone, 'beijing_time': moment.isoformat(),
                      'before_may6_open': bool(moment < pd.Timestamp('2019-05-06T09:30:00+08:00')),
                      'timezone_verified': False})
    save('results/推文档案时区敏感性.json', {'assumptions_only': True, 'scenarios': zones,
         'limit': '三种常见时区假设都落在复市前；并未证实档案实际时区或完整首次传播时钟。'})
    print('已核对11个信息节点；直接下载失败的正文保留为明确标注的网页事实摘记。')


def figure():
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    plt.rcParams['font.family'] = FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    data = pd.read_csv(OUT / 'results/完整逐日行情_波动来源与已知货币.csv', parse_dates=['date'])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10.2), sharex=True, gridspec_kw={'height_ratios': [2.7, 2.1, 1.0]})
    fig.patch.set_facecolor('#FAFBFC')
    for ax in axes:
        ax.set_facecolor('#FAFBFC')
        ax.grid(axis='y', color='#DEE3E9', linewidth=.7)
        ax.axvspan(pd.Timestamp('2019-05-01'), pd.Timestamp('2019-05-06'), color='#C8D0DA', alpha=.24)
    ax = axes[0]
    ax.plot(data.date, data.close, color='#233D59', linewidth=2.4, marker='o', markersize=4, label='510300收盘价')
    ax.scatter(pd.to_datetime(['2019-04-15', '2019-04-16']), [4.044, 3.938], marker='D', color='#9B650E', s=45, zorder=5, label='原E0与E1入场开盘')
    ax.scatter(pd.to_datetime(['2019-05-06']), [3.778], marker='s', color='#9B650E', s=42, zorder=5, label='5月6日复市开盘')
    labels = [
        ('2019-04-17', 4.067, '4/17 经济数据', (-23, 32)),
        ('2019-04-19', 4.120, '4/19 收盘后政策表述', (60, 20)),
        ('2019-04-30', 3.896, '4/30 PMI更新', (-50, 24)),
        ('2019-05-06', 3.673, '5/6 复市', (25, -25)),
        ('2019-05-09', 3.607, '5/9 盘中货币更新', (12, -17)),
        ('2019-05-15', 3.712, '5/15 经济更新\nE0结束', (-30, 31)),
    ]
    for day, value, label, offset in labels:
        ax.annotate(label, (pd.Timestamp(day), value), xytext=offset, textcoords='offset points',
                    fontsize=10, color='#324459', arrowprops={'arrowstyle': '-', 'color': '#8D99A8', 'lw': .8})
    ax.text(pd.Timestamp('2019-05-03'), 4.19, '五一休市\n5/5关税表态', ha='center', va='top', fontsize=10, color='#596778')
    ax.set_ylim(3.51, 4.235)
    ax.set_ylabel('ETF价格（元）')
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    axes[1].plot(data.date, data.rv20_recomputed * 100, color='#187B85', lw=2.3, label='20日总波动率')
    axes[1].plot(data.date, data.d20_recomputed * 100, color='#AD493D', lw=2.3, label='20日下行波动尺度')
    axes[1].set_ylabel('年化尺度（%）')
    axes[1].legend(loc='upper left', frameon=False, fontsize=10)
    axes[1].annotate('4/19→4/30：总波动下降，下行尺度上升',
                     (pd.Timestamp('2019-04-30'), 20.4056), xytext=(-208, -30), textcoords='offset points',
                     fontsize=10, arrowprops={'arrowstyle': '->', 'color': '#667385'})
    money_times = pd.to_datetime(['2019-04-12 15:00:00', '2019-04-12 16:00:03', '2019-05-09 10:00:03', '2019-05-16 15:00:00'])
    before = data.loc[data.date.eq('2019-04-12'), 'close_known_money_spread_pp'].iloc[0]
    axes[2].step(money_times, [before, -4, -5.6, -5.6], where='post', color='#665493', linewidth=2.3)
    axes[2].set_ylabel('当时公布的\nM1−M2（百分点）')
    axes[2].set_ylim(-6.4, -3.35)
    axes[2].text(pd.Timestamp('2019-04-21'), -3.83, '3月数据：−4.0', ha='center', fontsize=10)
    axes[2].text(pd.Timestamp('2019-05-12'), -5.35, '4月数据：−5.6', ha='center', fontsize=10)
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[2].set_xlim(pd.Timestamp('2019-04-11 12:00'), pd.Timestamp('2019-05-17'))
    fig.suptitle('2019年原观察窗口：货币信息、指数价格与波动并非同步变化', x=.08, y=.972, ha='left', fontsize=19, color='#233D59')
    fig.text(.08, .932, '原E0：4/15开盘至5/15收盘 −8.21%　｜　原E1：4/16开盘至5/16收盘 −5.28%', fontsize=12, color='#586678')
    fig.text(.08, .043, '资料：原510300日线与现金分红；央行、统计局及公开政策原文。所有日期沿用原20交易日窗口。', fontsize=9, color='#5D6977')
    fig.text(.08, .021, '历史路径核对；消息标注不构成因果归因。网页时钟为现存页面重建，非历史不可变快照。5/5推文档案未标时区。', fontsize=9, color='#5D6977')
    fig.subplots_adjust(left=.085, right=.965, bottom=.095, top=.90, hspace=.19)
    fig.savefig(OUT / 'figures/2019年指数信息与波动时间线.png', dpi=180)
    fig.savefig(OUT / 'figures/2019年指数信息与波动时间线.svg')
    plt.close(fig)
    print('已生成完整价格、波动与已知货币信息的三层时间线。')


def report():
    segments = pd.read_csv(OUT / 'results/原窗口_固定股数分段贡献.csv')
    macro = pd.read_csv(OUT / 'results/两期货币_各自公布后才可用.csv')
    source_lookup = {s['id']: s['url'] for s in json.loads((OUT / 'source_receipts.json').read_text(encoding='utf-8'))}
    source_lookup['rrr_pbc_reprint'] = 'https://dfjrjgj.hunan.gov.cn/dfjrjgj/tslm_71665/mtjj/201905/t20190507_5328854.html'
    labels = ['原开盘→4月19日收盘', '4月19日收盘→4月30日收盘', '4月30日收盘→5月6日开盘',
              '5月6日开盘→当日收盘', '5月6日收盘→5月9日收盘', '5月9日收盘→原末日收盘']
    path_rows = []
    for i, label in enumerate(labels):
        a = segments[segments.clock == 'E0'].iloc[i]
        b = segments[segments.clock == 'E1'].iloc[i]
        path_rows.append(f'| {label} | {a.return_contribution_on_original_capital * 100:+.2f} | {b.return_contribution_on_original_capital * 100:+.2f} |')
    credit_rows = []
    for title, field in [('剪刀差水平（百分点）', 'spread_pp'), ('剪刀差三个月变化（百分点）', 'delta3_spread_pp'),
                         ('近三个月企业贷款同比多增（亿元）', 'credit_3_corporate_total_yoy_change_yi'),
                         ('其中企业中长期同比多增（亿元）', 'credit_3_corporate_long_yoy_change_yi'),
                         ('其中企业短期同比多增（亿元）', 'credit_3_corporate_short_yoy_change_yi'),
                         ('其中票据融资同比多增（亿元）', 'credit_3_bills_yoy_change_yi')]:
        a, b = macro.iloc[0][field], macro.iloc[1][field]
        credit_rows.append(f'| {title} | {a:+,.1f} | {b:+,.1f} |')
    txt = f'''# 2019年510300货币信用与信息更新复盘

这份复盘沿用原来的2019年3月病例，以510300的完整价格路径为主线。能够确认的是：**剪刀差改善、信贷同比多增和波动下降，并没有在这个病例中共同构成稳定的上涨环境。入场前上涨已发生，信用多增的结构偏票据和短贷，降波又包含旧下跌退出窗口；持有期间还陆续出现新的经济、政策和外部信息。**这些事实需要一起解释。

原E0为4月15日开盘至5月15日收盘，20个交易日收益−8.21%；延迟一天的原E1为4月16日开盘至5月16日收盘，收益−5.28%。两种观察都保留，均为固定股数含应享现金的毛收益，没有费用和完整账户约束。两个窗口高度重叠，只是同一病例的时钟敏感性。

![2019年指数信息与波动时间线](figures/2019年指数信息与波动时间线.png)

## 货币改善的来源与当时已经发生的价格变化

4月12日收盘后公布的3月数据是M1同比4.6%、M2同比8.6%，剪刀差−4.0个百分点，比三个月前改善2.6个百分点。沿用已完成的余额核对：当期M1相对M2的三个月余额变化贡献−4.12个对数百分点，基数及修订合并项贡献+6.67，合计约+2.55个对数百分点。这是同比相对变化的对数分解，不能把它与普通的+2.6个百分点直接混加；它说明当期相对余额并没有随同比改善同步增强，且未经季节调整，不能把全部基数及修订项当成经济原因。[央行一季度金融统计]({source_lookup['money_march']})

信用结构进一步限定了“改善”的含义：近三个月企业贷款同比多增13900亿元，其中票据多增8380亿元、短贷多增4513亿元，中长期仅多增1200亿元；企业总量与分项之间仍保留−193亿元残差，不把分项强行凑平。企业和住户中长期都多增，提供了需求相关支持，但总量多增并不等于全部来自长期投资需求，更不等于沪深300整体盈利会按相同比例增加。票据和短贷也可能服务真实经营，不能仅凭期限就认定空转或无效。

与此同时，510300此前60个交易日已经上涨29.50%。这只能确认价格先于本次公告上涨，尚不能证明此前上涨究竟有多少由货币预期、政策或其他因素引起。把已出现的宏观改善直接当成尚未反映的新利好，会跳过这个定价位置问题。

## 原窗口内的条件持续变化

| 公开时间或当前可核实范围 | 新增事实 | 对解释的限制 |
|---|---|---|
| 4月12日16:00:03 | 3月货币及一季度信用公布 | 4月12日收盘还不知道，原观察快照21:00才包含它 |
| 4月17日10:00 | 一季度GDP初步同比6.4%，3月工业增加值同比8.5% | 未冻结事前一致预期，不能量化“超预期” |
| 4月19日17:23:19 | 经济会议提出结构性去杠杆、财政加力提效、货币松紧适度 | 是收盘后信息；原文没有宣布加息或全面收紧 |
| 4月30日09:00 | 制造业PMI由50.5回落到50.1，新订单由51.6到51.4 | 边际放缓与仍在扩张区间同时存在 |
| 5月5日，推文档案时区未认证 | 美方提出提高2000亿美元清单关税的意图 | 常见UTC、美国东部、西部时区假设都在5月6日开盘前，但仍不是认证首发时钟 |
| 5月7日15:05的央行问答转载 | 县域农商行较低准备金率，约2800亿元用于民营和小微信贷 | 不能把转载时间冒充最早公告；也不能把结构性银行资金自动视为指数资金流入 |
| 5月9日10:00:03 | 4月M1同比2.9%、M2同比8.5%，剪刀差−5.6 | 按现存央行网页重建为盘中更新，5月9日开盘仍使用3月数据 |
| 5月9日美方法律通知，5月10日生效 | 部分关税从10%升到25% | 法律实施与5月5日政策表态是不同节点 |
| 5月13日中方公告，6月1日生效 | 对部分美国商品调整关税 | 原窗口内只有宣布，实际实施尚未发生 |
| 5月15日10:00 | 4月工业增加值同比5.4%、社零7.2% | 同稿给出的节假日因素调整后社零测算为8.7%，不能只用7.2%讲需求全面恶化 |

经济、PMI及政策事实分别见[统计局一季度发布]({source_lookup['economy_q1']})、[4月PMI]({source_lookup['pmi_april']})、[4月经济发布]({source_lookup['economy_april']})、[4月19日会议原文]({source_lookup['politburo']})、[央行降准问答政府转载]({source_lookup['rrr_pbc_reprint']})。关税的表态、法律及回应分别见[5月5日原推文档案]({source_lookup['tariff_intent']})、[美方法律通知]({source_lookup['tariff_notice']})、[美方5月10日声明]({source_lookup['ustr_may10']})、[中方5月13日公告]({source_lookup['china_tariff']})。这些是历史消息顺序，不是完整事件清单或新闻冲击估计。

5月9日的更新尤其说明，**仍然为正的“三个月剪刀差改善”，可以与企业中长期信用转弱并存**：

| 分别在各公告后才可使用的观测 | 4月12日公布的3月数据 | 5月9日公布的4月数据 |
|---|---:|---:|
{chr(10).join(credit_rows)}

同一份4月报告还显示，当月企业存款减少1738亿元、住户存款减少6248亿元，财政性存款增加5347亿元。它们提示需要检查存款部门迁移、财政收支与信用结构；缺少活期定期、支付和税款流向的完整连接，尚不能把M1放缓全部归给缴税，也不能从这些部门存款变化直接推出资金进入或退出股票市场。[央行4月金融统计]({source_lookup['money_april']})

## 亏损在什么时候发生

以下是对**原投入的收益贡献，单位为百分点**。各段都用各自原入场价作分母，能够相加；不是逐段独立复利收益，也不是某条新闻的因果影响。

| 完整原窗口中的分段 | E0贡献 | E1贡献 |
|---|---:|---:|
{chr(10).join(path_rows)}
| 全部原20交易日 | −8.21 | −5.28 |

4月30日收盘时，E0累计已经−3.66%，E1已经−1.07%。因此，把整个20日亏损都解释为五一假期后的关税消息，时间上不成立。5月6日从前一收盘到当日收盘本身下跌5.72%，在E0原投入上贡献−5.51个百分点，其中跳空−2.92、盘中−2.60。跳空存在信息到来后已经体现在首个交易价的部分，但本轮没有检验任何即时执行或退出规则。

5月9日收盘到各自原终点则修复了一部分跌幅，E0贡献+2.60、E1贡献+3.12个百分点。因此也不能反向把“5月9日数据偏弱”写成此后每日下跌的机械解释。消息、预期和价格反应之间仍缺少唯一可识别的因果分配。

## 波动下降为何没有代表风险消退

在原4月12日观察点，下行波动尺度已从五个交易日前的16.21%降到11.97%。但新进入五日的负收益平方和为0.000350，前一组五日仅0.000003，退出20日窗口的旧五日为0.001299。下降主要是更大的旧下跌离开窗口，近期负收益能量却比紧邻的前五日增强。这个算术关系当时即可计算，不能据此宣称已证明一种新预测规则。

同一组3月货币背景持续期间，4月19日至30日总波动率从25.09%降到20.41%，下行尺度从11.95%升到14.40%，价格同时下跌5.44%。较大的上涨或下跌退出滚动窗口，都可能降低总波动；总波动下降与下行压力缓解必须分别检查。4月24日甚至出现下行尺度五日下降，但最近五日负收益平方和仍高于紧邻的前五日，完整逐日表保留了这一过程。

因此，这个病例中可确认的联系是：信用改善的来源并不单一，价格已先涨一段，滚动降波不完全代表新风险减弱，之后又有新的宏观与政策信息。上述联系可以解释原先平面标签遗漏了什么，尚未给出可跨时期复现的买卖优势。对指数的下一层检验仍应落在整体盈利预期、折现条件、风险溢价和已发生价格反应之间，个股资料只在确有助于核对指数内部传导时补充。

## 范围与复算

保留全部104个月原输入，新增这一个既有病例的22条交易日路径、44个开盘收盘货币时钟节点、两套原20日收益及12段贡献。检查原收益重算、现金权益、分段求和、下行平方窗口恒等式以及5月9日盘中更新；没有新模型、参数搜索、账户回测或新增个股研究。

来源精度须与结论一起保留：现存网页日期不是历史不可变版本。美国总统档案时区未认证；降准政府网旧链接404，央行问答转载由网络工具读到但直接TLS下载失败；财政部公告由网络工具读到但直接下载502。后二者保留明确标注的结构化事实摘记，没有伪装成已保存原始HTML。美国法律文件的备案行不当作已认证公开时刻，采用公报日期的美国东部日终上界。以上缺口不影响价格分段算术，但限制最早消息到达与即时反应的研究。

本轮完成的是历史机制复盘，目标状态仍为进行中。不能把一个已知失败病例的解释，升级为独立验证或未来确定上涨结论。
'''
    (OUT / '2019年指数信息更新与波动复盘.md').write_text(txt, encoding='utf-8')
    print('已保存结合信用来源、已发生定价与后续新信息的指数复盘。')


def complete():
    check = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    assert check['status'] == 'PASS_FIXED_WINDOW_ARITHMETIC_AND_CLOCKS'
    needed = ['source_content_review.json', 'results/信息时间线_事实与解释边界.csv',
              'figures/2019年指数信息与波动时间线.png', '2019年指数信息更新与波动复盘.md']
    assert all((OUT / name).is_file() for name in needed)
    for script in ['index_information_sequence_v27.py', 'publish_index_information_v27.py']:
        shutil.copy2(ROOT / 'research' / script, OUT / 'code' / script)
    save('completion.json', {'at': now(), 'status': 'COMPLETED_INDEX_INFORMATION_SEQUENCE',
         'goal_turn_classification': 'PROGRESS', 'goal_achieved': False,
         'result': '原窗口内宏观信息会更新；假期前已有亏损；滚动总波动下降与下行压力上升可并存。',
         'original_inputs_retained': True, 'independent_validation': False, 'causal_identification': False,
         'no_candidate_promoted': True, 'visual_review': '已查看生成PNG，中文、曲线与标注可读',
         'report_sha256': hashlib.sha256((OUT / needed[-1]).read_bytes()).hexdigest()})
    print('本轮历史指数信息顺序核对已完成；整体研究目标继续保持进行中。')


if __name__ == '__main__':
    modes = {'sources': sources_and_events, 'figure': figure, 'report': report, 'complete': complete}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/publish_index_information_v27.py sources|figure|report|complete')
    modes[sys.argv[1]]()
