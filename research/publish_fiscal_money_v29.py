"""交付指数层面的财政、货币结构与既有价格波动证据。"""
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
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_fiscal_money_transmission_v29'
MONTHS = [f'2025-{m:02d}' for m in range(6, 11)]
H1_URL = 'https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202507/t20250725_3968586.htm'
Q3_URL = 'https://m.mof.gov.cn/czxw/202510/t20251017_3974415.htm'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows])


def read_results():
    names = ['五月基点及六月至十月_事后原表余额.csv', '六月至十月_逐月货币与政府债权存款变化.csv',
             '五个原观察点_当时财政与后来同月资料.csv', '两本预算_累计及跨原公告推导单月.csv']
    return [pd.read_csv(OUT / 'results' / name).set_index('stat_month') for name in names]


def source_facts():
    h1_text = (OUT / 'sources/mof_h1_press.txt').read_text(encoding='utf-8')
    for phrase in ['已发行1.8万亿元', '已使用1.44万亿元', '上半年支出了2.43万亿元', '5000亿元']:
        assert phrase in h1_text, phrase
    facts = [
        ['h1_swap_2025_issue', '2025-07-25', '截至2025年6月末', '2025年额度置换债已发行', 18000, '发行金额', H1_URL, 'SAVED_HTML_REVIEWED'],
        ['h1_swap_2025_used', '2025-07-25', '截至2025年6月末', '2025年额度置换债已使用', 14400, '使用金额', H1_URL, 'SAVED_HTML_REVIEWED'],
        ['h1_swap_issue_less_use', '2025-07-25', '截至2025年6月末', '同额度发行减使用，算术差', 3600, '发行使用差，不定位存放账户', H1_URL, 'DERIVED_FROM_SAVED_HTML'],
        ['h1_swap_2024_used', '2025-07-25', '截至2025年上半年末', '2024年额度已使用完毕', 20000, '累计完成，不等于全额在2025年使用', H1_URL, 'SAVED_HTML_REVIEWED'],
        ['h1_bond_fund_spending', '2025-07-25', '2025年上半年', '所列债券资金支出', 24300, '含地方专项、超长期及银行注资债券资金', H1_URL, 'SAVED_HTML_REVIEWED'],
        ['h1_bank_recapitalization', '2025-07-25', '2025年上半年', '特别国债支持大行核心一级资本补充', 5000, '金融资本补充，不能等同商品采购', H1_URL, 'SAVED_HTML_REVIEWED'],
        ['q3_bond_fund_spending', '2025-10-17', '2025年前三季度', '所列债券资金支出', 42100, '包含以往年度结转债券支出', Q3_URL, 'WEB_TEXT_REVIEWED_RAW_HTML_NOT_SAVED'],
        ['q3_local_debt_quota', '2025-10-17', '发布会所述近期', '下达地方债务结存限额', 5000, '额度下达，地方在办理程序；未提供已使用额', Q3_URL, 'WEB_TEXT_REVIEWED_RAW_HTML_NOT_SAVED'],
    ]
    f = pd.DataFrame(facts, columns=['fact_id', 'published_date', 'reference_period', 'fact', 'amount_yi', 'measurement_stage', 'source_url', 'source_status'])
    f['available_at_upper_bound'] = f.published_date + 'T23:59:59+08:00'
    f['first_historical_vintage_verified'] = False
    csv('财政用途及进展_逐项事实与公开日期.csv', f)
    receipts = json.loads((OUT / 'new_source_receipts.json').read_text(encoding='utf-8'))
    q3 = next(x for x in receipts if x['id'] == 'mof_q3_press')
    assert q3['status'] == 'SOURCE_NOT_SAVED'
    save('sources/mof_q3_press.web-reviewed-facts.json', {
        'at': now(), 'source_url': Q3_URL, 'title': '2025年前三季度财政收支情况新闻发布会文字实录',
        'published_date': '2025-10-17', 'method': 'web.open读取并人工核对官方全文；本文件仅保存结构化事实，不是原始HTML或全文镜像',
        'raw_http_receipt': q3, 'facts': f[f.fact_id.str.startswith('q3_')].to_dict('records'),
        'source_sections_checked': ['债券资金支出及往年结转范围', '地方政府债务结存限额下达与后续发行使用程序'],
        'q3_debt_swap_actual_use_amount_disclosed_in_selected_source': False,
        'historical_first_version_authenticated': False,
    })
    _, _, m, _ = read_results()
    clock_rows = []
    for month, r in m.iterrows():
        for x in f.to_dict('records'):
            clock_rows.append({'stat_month': month, 'original_snapshot': r.snapshot_at,
                               'fact_id': x['fact_id'], 'fact_available_at': x['available_at_upper_bound'],
                               'available_by_original_snapshot': pd.Timestamp(x['available_at_upper_bound']) <= pd.Timestamp(r.snapshot_at),
                               'first_historical_vintage_verified': False})
    csv('五个观察点_财政用途事实是否已公开.csv', pd.DataFrame(clock_rows))
    save('scope_clarifications.json', {
        'at': now(), 'unchanged_selection': MONTHS,
        'meaning_of_continued_improvement': '五个观察点的剪刀差三个月变化均为正；不是五个月逐月改善。10月相对9月恶化0.8个百分点。',
        'new_scope': '9至10月分项余额、同阶段财政执行及资料公布衔接；6至8月余额来自V8，原收益及对数分解直接复用。',
        'no_new_return_hypothesis_or_threshold': True,
    })
    print('已保存财政用途事实及40条原观察点公开时间对应关系。')


def figure():
    _, d, m, _ = read_results()
    plt.rcParams.update({'font.family': FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus': False, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(13.3, 12.0), sharex=True, gridspec_kw={'height_ratios': [1.2, 1, 1]})
    fig.patch.set_facecolor('#FAFBFC')
    x = np.arange(5)
    for ax in axes:
        ax.set_facecolor('#FAFBFC')
        ax.grid(axis='y', color='#DEE5EB', zorder=0)
        ax.axhline(0, color='#677785', linewidth=.8)
        ax.set_xlim(-.6, 4.6)
    positive, negative = np.zeros(5), np.zeros(5)
    for values, label, color in [(d.corporate_demand.to_numpy() / 10000, '单位活期', '#345F7C'),
                                  (d.personal_demand_derived.to_numpy() / 10000, '个人活期（推导）', '#C57D45'),
                                  ((d.m0 + d.payment_reserves + d.m1_identity_error).to_numpy() / 10000, '现金、支付备付金及舍入', '#95A9B7')]:
        bottom = np.where(values >= 0, positive, negative)
        axes[0].bar(x, values, bottom=bottom, width=.55, color=color, label=label, zorder=3)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axes[0].plot(x, d.m1_table / 10000, 'D-', color='#263443', markersize=5, linewidth=1.4, label='M1净变化', zorder=5)
    axes[0].set_ylim(-3.7, 6.6)
    axes[0].set_ylabel('单月余额变化（万亿元）')
    axes[0].set_title('A  事后分项：9月M1增加，单位活期减少；10月个人活期反向变化', loc='left', fontsize=13, pad=12)
    axes[0].legend(loc='upper right', ncol=2, fontsize=10, frameon=False)
    for i, value in enumerate(d.m1_table / 10000):
        axes[0].text(i, value + (.2 if value >= 0 else -.24), f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', color='#263443')
    axes[1].plot(x, m.past_return60 * 100, 'o-', color='#345F7C', label='此前60交易日收益')
    axes[1].plot(x, m.rv20 * 100, 's-', color='#C57D45', label='20日总波动率（年化）')
    axes[1].plot(x, m.downside20 * 100, '^-', color='#758777', label='20日下行尺度（年化）')
    axes[1].set_ylim(0, 27)
    axes[1].set_ylabel('百分比；收益与波动定义不同')
    axes[1].set_title('B  货币公告快照时：510300已经上涨，风险路径也在变化', loc='left', fontsize=13, pad=12)
    axes[1].legend(loc='upper left', ncol=3, fontsize=10, frameon=False)
    for col, offset, color, label in [('E0_20_return', -.16, '#345F7C', '原E0'), ('E1_20_return', .16, '#C57D45', '延迟一天E1')]:
        values = m[col].to_numpy() * 100
        axes[2].bar(x + offset, values, width=.29, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            axes[2].text(i + offset, value + (.3 if value >= 0 else -.3), f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
    axes[2].set_ylim(-4.2, 12.3)
    axes[2].set_ylabel('后20交易日观察毛收益（%）')
    axes[2].set_title('C  原后续窗口：相同的信用少增背景，没有给出统一的指数方向', loc='left', fontsize=13, pad=12)
    axes[2].legend(loc='upper left', ncol=2, frameon=False)
    labels = [month + '\n公告快照 ' + m.loc[month, 'snapshot_at'][5:10] for month in MONTHS]
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel('货币统计月与原公告快照日期（均为2025年，非同一天）', labelpad=10)
    fig.suptitle('把货币结构、已经发生的指数行情与后续结果放回各自时点', x=.095, y=.98, ha='left', fontsize=18, color='#263F57')
    fig.text(.095, .946, '完整保留2025年6至10月。剪刀差三个月变化均为正，但10月相对9月已恶化0.8个百分点。', fontsize=11, color='#516171')
    fig.text(.095, .045, 'A使用12月网址版本年度表，全部为事后解释；B、C复用原104个月底表。月度余额未作季节调整。', fontsize=10, color='#516171')
    fig.text(.095, .024, '五个月属于一个连续阶段，窗口可能重叠。E0/E1未计成本及账户约束；图形不识别资金去向或涨跌原因。', fontsize=10, color='#516171')
    fig.subplots_adjust(left=.095, right=.965, top=.90, bottom=.13, hspace=.43)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / f'figures/货币持有结构与510300前后路径.{suffix}', dpi=180)
    plt.close(fig)
    print('已生成货币分项、指数已走行情和原后续结果的三层图。')


def report():
    b, d, m, fiscal = read_results()
    total = json.loads((OUT / 'results/五个月余额变化合计.json').read_text(encoding='utf-8'))
    m1_rows, context_rows, base_rows, fiscal_rows, clock_rows = [], [], [], [], []
    for month in MONTHS:
        a, r, f = d.loc[month], m.loc[month], fiscal.loc[month]
        m1_rows.append([month, f'{a.m1_table:+,.2f}', f'{a.corporate_demand:+,.2f}', f'{a.personal_demand_derived:+,.2f}',
                        f'{a.m0 + a.payment_reserves:+,.2f}', f'{a.m1_identity_error:+.2f}'])
        context_rows.append([month, r.snapshot_at[5:10], f'{r.delta3_spread_pp:+.1f}', f'{r.credit_3_corporate_long_yoy_change_yi:+,.0f}',
                             f'{r.past_return60:.2%}', f'{r.rv20:.2%}', f'{r.downside20:.2%}', f'{r.E0_20_return:+.2%}', f'{r.E1_20_return:+.2%}'])
        base_rows.append([month, f'{r.spread_pp:+.1f}', f'{r.delta3_spread_pp:+.1f}', f'{r.d3_relative_current_log_pp:+.3f}',
                          f'{r.d3_relative_base_revision_log_pp:+.3f}', f'{r.d3_relative_current_log_pp + r.d3_relative_base_revision_log_pp:+.3f}'])
        fiscal_rows.append([month, f'{f.general_expenditure_month_yi:,.0f}', f'{f.fund_expenditure_month_yi:,.0f}',
                            f'{f.general_ytd_yoy_pp:+.1f}%', f'{f.fund_ytd_yoy_pp:+.1f}%', f'{a.cb_gov_deposits:+,.2f}'])
        clock_rows.append([month, r.snapshot_at[5:10], r.latest_fiscal_month_at_original_snapshot,
                           r.latest_fiscal_available_at[5:10], r.same_stat_month_fiscal_available_at[5:10], '否'])
    counterpart = pd.read_csv(OUT / 'results/五个月_M2全部资产负债对应项.csv').groupby('item', sort=False).m2_arithmetic_contribution_yi.sum()
    names = {'nfa': '国外净资产', 'government_net': '对政府净债权', 'nonfinancial_claims': '对非金融部门债权',
             'other_financial_claims': '对其他金融部门债权', 'excluded_deposits': '不纳入M2的存款（负号）',
             'bonds': '债券负债（负号）', 'capital': '实收资本（负号）', 'other_net': '其他净项（负号）', 'counterpart_rounding': '原表舍入'}
    counter_rows = [[names[k], f'{v:+,.2f}'] for k, v in counterpart.items()]
    counter_rows.append(['M2余额净变化', f'{total["m2"]:+,.2f}'])
    text = f'''# 510300：财政用途、货币持有结构与指数定价

**这轮更明确的结论是：2025年6至10月的货币改善包含不同部门、不同用途与同比基数的变化。9月M1余额增加约1.92万亿元，单位活期却减少约0.16万亿元，个人活期增加约1.80万亿元。这不足以代表企业经营需求同步增强，更不能把货币净增加直接换算成510300的新增买盘。**

对510300，应同时辨别总需求与盈利预期、融资和债务压力、贴现率及风险偏好、价格已经反应的程度。化债、银行资本补充和实际采购可能通过不同渠道影响指数；现有资料能够确认这些用途并存，不能确定各渠道对指数涨幅的因果份额。

本轮的测量新增是9至10月存款分项、五个月财政执行与公布日期衔接，以及两次财政发布会的用途进展。6至8月银行余额、原收益和对数分解均复用已有结果。完整104个月、741列原底表保留，五个月不是新筛出的交易信号，也不是五次独立验证。

**“连续改善”的准确含义需要先讲清楚。**

这五个观察点的剪刀差相对三个月前都改善，但10月为−2.0个百分点，比9月−1.2恶化0.8个百分点。“三个月变化为正”与“本月继续改善”不同。原公告的分解如下：

{md_table(['货币统计月', '剪刀差水平（百分点）', '三个月变化（百分点）', '当期相对余额项（对数百分点）', '隐含基数及修订项（对数百分点）', '后两项合计'], base_rows)}

对数分解由各次公告余额和同比反推相容基数。它含舍入及潜在修订，不是独立认证的上年同版本余额；合计也不等于普通百分点三个月变化。6、9、10月的当期相对余额项为负，7、8月虽为正，基数及修订项仍更大。这是沿用的既有发现，不能把本轮财政资料说成新识别了基数原因。

**新M1增长由谁持有，影响它能支持哪一种解释。**

以下均是2025年12月网址版本银行年度表中的月末余额差，单位亿元，仅供事后解释。个人活期由存款性公司活期总额减其他存款性公司单位活期推导；现金与支付备付金单列，保留原表舍入。

{md_table(['统计月', 'M1变化', '单位活期变化', '个人活期变化（推导）', '现金及支付备付金', '舍入残差'], m1_rows)}

9月的个人活期增加不能自动解释成准备买股票，10月个人活期减少也不能自动解释成股市资金流出。这些是部门净余额，缺少逐笔转移关系，而且未经季节调整。原表的“单位活期”也不等同于企业订单、利润或新增投资。月度余额分项解释总量的组成，不能替代同比基数分解。[央行存款性公司概览](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172611493.pdf)、[其他存款性公司资产负债表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172647559.pdf)

**财政融资确实包含不同用途，且发行与使用有时间差。**

财政部7月25日披露，2025年额度置换债截至6月底发行1.8万亿元、使用1.44万亿元，两者相差3600亿元。2024年额度则已累计使用完毕，不能把该额度全部记为2025年新增使用。同期所列债券资金支出2.43万亿元包含银行注资等用途，5000亿元特别国债支持四家大行补充核心一级资本。[财政部上半年发布会]({H1_URL})

10月17日披露的前三季度相关债券资金支出4.21万亿元，明确含往年结转；另有5000亿元地方债务结存限额下达，当时仍在办理发行使用程序。这项额度与上半年银行注资是不同事项，不能把二者相加称为已投入新增需求。[财政部前三季度发布会]({Q3_URL})

因此，使用已发行资金偿还旧债、补充金融资本、清偿旧欠款与购买新商品服务，不能使用同一种需求解释；缓解偿债压力可以影响风险预期，也不必立即表现为新一轮企业中长期贷款多增。这里是机制解释，尚未识别其净效果。央行“实收资本”科目与核心一级资本的范围也不同，不用它强行配平上述注资。

**实际财政支出与政府存款净增加，可以同时发生。**

{md_table(['统计月', '一般公共预算推导单月支出（亿元）', '政府性基金推导单月支出（亿元）', '一般预算累计同比', '基金预算累计同比', '央行政府存款月变动（亿元）'], fiscal_rows)}

支出月度金额由相邻原累计公告相减得到，可能包含跨公告修订；原始公告、累计额和收入分项均保留。单月金额未季调，不能仅凭高低判断力度转折。两本预算之间存在调入调出、结转等关系，收支差不等于财政赤字，也不能与央行政府存款机械配平。

从5月末至10月末，其他存款性公司对政府债权增加{total['odcs_gov_claims']:,.2f}亿元，央行对政府债权变化{total['cb_gov_claims']:+,.2f}亿元，央行政府存款增加{total['cb_gov_deposits']:,.2f}亿元。政府净债权的恒等式为：48,681.30 − 2,903.61 − 3,749.21 − 0.02（舍入）= 42,028.46亿元。

这不能解释成“财政向市场净投放4.20万亿元”，也不能把政府存款净增加说成同额发债没有使用。债权、政府存款与预算支出的统计对象和毛流量范围不同。[央行货币当局资产负债表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172659799.pdf)

为避免只挑政府这一侧，同期M2的全部资产负债对应项如下。表中是算术贡献，不是独立因果贡献。

{md_table(['对应项', '对M2净变化的算术贡献（亿元）'], counter_rows)}

持有人一侧，纳入M2的其他金融性公司存款同期增加{total['odcs_financial_deposits_m2']:,.2f}亿元；这是金融部门余额变化，缺少账户与交易用途，不能命名为股市增量资金。对非金融部门债权的净增也不等于企业中长期贷款同比多增，二者覆盖对象和比较基准不同。

**历史判断只能使用当时已公布的那一期财政信息。**

{md_table(['货币统计月', '货币公告后21点快照', '当时最近财政统计月', '该财政资料公布日', '同货币统计月财政后来公布日', '同月财政在原起点已知'], clock_rows)}

五个货币起点都只能使用此前一期财政执行。同为“6月数据”的上半年财政进展，7月25日才公布，不能进入7月14日；同为“9月数据”的前三季度进展，10月17日才公布，不能进入10月15日。日期没有精确时刻的财政来源沿用日终上界。年度银行表均为事后版本，不进入这些原起点。

**把这些证据放回指数，最直接的对照是价格已经走了多少，以及剩余路径。**

{md_table(['货币统计月', '快照日', '剪刀差三月变化', '企业中长期近三月同比多增（亿元）', '此前60日收益', 'RV20年化', 'D20年化', '后20日E0', '后20日E1'], context_rows)}

企业中长期列的负数是贷款净增加额相对上年同期减少，不是贷款余额收缩。五个月信用持续少增时，7月统计数据公布前510300已涨8.90%，原E0后20日仍涨6.40%；8月数据公布前已涨19.16%，之后E0仅涨0.39%；10月数据对应的原E0、E1都为负。这些差异说明同一种信用方向不能替代指数判断。它们不证明涨多必跌，也不证明哪一种资金导致上涨。

原快照总波动率从9月12日的21.33%降至10月15日的17.88%、11月13日的15.68%，下行尺度分别10.65%、10.42%、10.37%。三个时点之间的总波动下降明显大于下行尺度下降；各自后续原E0既有正也有负。因此“总波动降低”不能直接命名风险已经消退或下一段上涨得到确认。这里比较的是原快照，未假定期间每天单调变化。

![完整五个月的货币结构与指数前后路径](<{(OUT / 'figures/货币持有结构与510300前后路径.png').as_posix()}>)

E0是公告可用后的原21点快照、下一交易日开盘至第20个交易日收盘；E1入场及终点均顺延一个交易日。现金分红按原固定份额权益近似处理，未计费用与完整账户约束。RV20来自20日总波动，D20按全部20日负收益平方年化，不是只以下跌日为分母。两套窗口和相邻月份可能重叠，不据此计算独立命中率。

**能够保留的指数结论与仍未解决的问题。**

现有证据支持把这段行情理解为多个宏观过程并存：同比基数支持剪刀差、活期持有结构变化、政府融资及偿债用途、实际财政支出，与指数已经发生的上涨和波动变化同时出现。企业信贷少增不能推出指数必须下跌，M1回升不能推出经营恢复，降波也不能单独推出安全上涨。

仍未识别的是：这些变化中有多少在原起点超出预期、哪些已反映在指数价格、还有多少独立的未来20日增量。当前资料尚未形成经独立验证、可用于前瞻判断的指数规律，不能将解释性成果视为总目标完成。本轮也不根据这五个月结果增加阈值、拟合新模型或重启已失败的财政预测模型。

复核覆盖原104个月741列、六份财政月报、三张年度银行表及全部五个月差分。余额恒等式保留0.01至0.02亿元级舍入；原价格收益未重选。前三季度发布会经网页工具全文核对，直接HTTP取回失败，保存的是明确标注的结构化事实，未冒充原始HTML。全部历史资料均未取得不可修订首发版本认证，未完成独立验证；本轮模型数、账户数均为0。

复算入口为[数据脚本](<{(ROOT / 'research/fiscal_money_transmission_v29.py').as_posix()}>)和[报告脚本](<{(ROOT / 'research/publish_fiscal_money_v29.py').as_posix()}>)；已保存的主要结果为[五个月完整证据链](<{(OUT / 'results/五个月完整链_当时信用价格与事后财政货币.csv').as_posix()}>)、[财政用途与公开日期](<{(OUT / 'results/财政用途及进展_逐项事实与公开日期.csv').as_posix()}>)、[40条原观察点事实时钟](<{(OUT / 'results/五个观察点_财政用途事实是否已公开.csv').as_posix()}>)和[保存结果复核](<{(OUT / 'saved_output_verification.json').as_posix()}>).

生成时间：{now()}。
'''
    (OUT / '财政货币结构与指数定价.md').write_text(text, encoding='utf-8')
    print('已交付完整五个月的财政货币结构与指数定价报告。')


def verify():
    b, d, m, f = read_results()
    fr = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == fr['protocol_sha256']
    for source in fr['inputs']:
        assert digest(OUT / 'inputs' / source['name']) == source['sha256']
    for source in fr['copied_sources']:
        assert digest(OUT / 'sources' / source['name']) == source['sha256']
    original = pd.read_csv(OUT / 'inputs/monthly.csv').set_index('stat_month')
    assert original.shape == (104, 740)
    assert list(d.index) == MONTHS and list(m.index) == MONTHS
    comparable = [c for c in b.columns if c in d.columns and pd.api.types.is_numeric_dtype(b[c])]
    diff_error = float((d[comparable] - b[comparable].diff().loc[MONTHS]).abs().max().max())
    total_error = float((d[comparable].sum() - (b.iloc[-1][comparable] - b.iloc[0][comparable])).abs().max())
    m1_error = float((d.m1_table - d[['m0', 'corporate_demand', 'personal_demand_derived', 'payment_reserves', 'm1_identity_error']].sum(axis=1)).abs().max())
    counter = pd.read_csv(OUT / 'results/五个月_M2全部资产负债对应项.csv').groupby('stat_month').m2_arithmetic_contribution_yi.sum()
    m2_error = float((counter - d.m2).abs().max())
    assert max(diff_error, total_error, m1_error, m2_error) < 1e-7
    original_columns = ['E0_20_return', 'E1_20_return', 'past_return60', 'rv20', 'downside20', 'spread_pp', 'delta3_spread_pp']
    original_error = float((m[original_columns] - original.loc[MONTHS, original_columns]).abs().max().max())
    assert original_error < 1e-12
    for key in ['general_expenditure', 'fund_expenditure', 'general_revenue', 'fund_revenue']:
        assert abs(f.loc[MONTHS, key + '_month_yi'].sum() - (f.iloc[-1][key + '_ytd_yi'] - f.iloc[0][key + '_ytd_yi'])) < 1e-8
    clocks = pd.read_csv(OUT / 'results/五个观察点_财政用途事实是否已公开.csv')
    assert len(clocks) == 40
    assert not clocks[clocks.stat_month.eq('2025-06')].available_by_original_snapshot.any()
    assert not clocks[clocks.stat_month.eq('2025-09') & clocks.fact_id.str.startswith('q3_')].available_by_original_snapshot.any()
    assert not m.same_stat_month_fiscal_known_at_origin.any()
    assert np.isclose(original.loc['2025-10', 'spread_pp'] - original.loc['2025-09', 'spread_pp'], -.8)
    assert (m.delta3_spread_pp > 0).all()
    assert d.loc['2025-09', 'corporate_demand'] < 0 < d.loc['2025-09', 'm1_table']
    # 与已目视核对的原PDF单页数值对照，覆盖新增两月及差分基点。
    visual_values = [('2025-05', 'm2', 3257838.11), ('2025-05', 'corporate_demand', 520368.05),
                     ('2025-08', 'm1_table', 1112255.70), ('2025-08', 'corporate_demand', 533310.14),
                     ('2025-09', 'm1_table', 1131455.07), ('2025-09', 'corporate_demand', 531672.35),
                     ('2025-09', 'cb_gov_deposits', 53251.69), ('2025-10', 'm1_table', 1119962.73),
                     ('2025-10', 'corporate_demand', 531896.31), ('2025-10', 'cb_gov_deposits', 59509.66)]
    for month, key, value in visual_values:
        assert abs(b.loc[month, key] - value) < .005, (month, key)
    check = {'at': now(), 'status': 'PASS_SAVED_OUTPUT_RECOMPUTATION', 'original_months': 104, 'original_columns': 741,
             'phase_months': MONTHS, 'fiscal_reports': len(f), 'source_fact_clock_rows': len(clocks),
             'saved_diff_max_error_yi': diff_error, 'sum_to_endpoint_max_error_yi': total_error,
             'm1_additive_max_error_yi': m1_error, 'm2_counterpart_max_error_yi': m2_error,
             'reused_original_market_max_error': original_error, 'visually_checked_pdf_values': len(visual_values),
             'same_month_fiscal_available_at_origin_count': 0, 'historical_first_vintage_verified': False,
             'new_models': 0, 'new_accounts': 0, 'independent_validation': False, 'goal_achieved': False}
    save('saved_output_verification.json', check)
    print(json.dumps(check, ensure_ascii=False, indent=2))


def complete():
    check = json.loads((OUT / 'saved_output_verification.json').read_text(encoding='utf-8'))
    assert check['status'] == 'PASS_SAVED_OUTPUT_RECOMPUTATION'
    for name in ['fiscal_money_transmission_v29.py', 'publish_fiscal_money_v29.py']:
        shutil.copy2(ROOT / 'research' / name, OUT / 'code' / name)
    save('completion.json', {'at': now(), 'status': 'COMPLETE_BOUNDED_DIAGNOSTIC', 'goal_turn_classification': 'PROGRESS',
                             'report': '财政货币结构与指数定价.md', 'report_sha256': digest(OUT / '财政货币结构与指数定价.md'),
                             'figure_visually_reviewed': True, 'goal_achieved': False, 'new_models': 0, 'new_accounts': 0,
                             'new_information': ['9至10月持有结构差分', '五个月财政执行与原公告时钟', '财政资金用途和使用进度'],
                             'retained_limits': ['历史首版未认证', '同一连续阶段不是独立样本', '事后余额不作原决策输入', '没有稳定方向规则']})
    print('本轮指数机制核对已完成；总目标仍未达成。')


if __name__ == '__main__':
    modes = {'facts': source_facts, 'figure': figure, 'report': report, 'verify': verify, 'complete': complete}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/publish_fiscal_money_v29.py facts|figure|report|verify|complete')
    modes[sys.argv[1]]()
