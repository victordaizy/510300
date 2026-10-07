import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const report = path.join(root, 'reports', 'research', '510300_pressure_recovery_v1');
const data = JSON.parse(await fs.readFile(path.join(report, 'workbook_data.json'), 'utf8'));
const wb = Workbook.create();
const font = 'Microsoft YaHei';
const sheets = ['研究结论', '同步行情', '事件', '订单', '结果', '日期覆盖', '费用演示', '数据来源'];
for (const name of sheets) wb.worksheets.add(name);

function table(name, title, note, headers, rows, widths) {
  const sh = wb.worksheets.getItem(name);
  sh.showGridLines = false;
  sh.getRangeByIndexes(0, 0, Math.max(rows.length + 5, 8), headers.length).format.font = { name: font, size: 10, color: '#243247' };
  sh.getRange('A1').values = [[title]];
  sh.getRange('A1').format.font = { name: font, size: 16, bold: true, color: '#192B47' };
  sh.getRangeByIndexes(0,0,1,headers.length).format.rowHeight=29;
  sh.getRange('A2').values = [[note]];
  sh.getRange('A2').format.font = { name: font, size: 10, italic: true, color: '#52627A' };
  const header = sh.getRangeByIndexes(3, 0, 1, headers.length);
  header.values = [headers];
  header.format = { fill: '#243C5A', font: {name: font, size: 10, bold: true, color: '#FFFFFF'}, wrapText: true, rowHeight: 38 };
  if (rows.length) {
    const body = sh.getRangeByIndexes(4, 0, rows.length, headers.length);
    body.values = rows;
    body.format.rowHeight = 32;
    body.format.verticalAlignment = 'center';
    const dataTable=sh.tables.add(`A4:${columnName(headers.length)}${rows.length+4}`, true, `Table${sheets.indexOf(name)+1}`);
    dataTable.style='TableStyleLight1';
  }
  for (let i = 0; i < widths.length; i++) sh.getRangeByIndexes(0, i, Math.max(rows.length + 5, 8), 1).format.columnWidth = widths[i];
  header.format.verticalAlignment='center';
  header.format.horizontalAlignment='center';
  if (rows.length > 15) sh.freezePanes.freezeRows(4);
  return sh;
}

function columnName(number) {
  let result = '';
  while (number) { const d = (number - 1) % 26; result = String.fromCharCode(65 + d) + result; number = Math.floor((number - 1) / 26); }
  return result;
}

const summary = table('研究结论', '510300 价格让步与压力恢复', '2026-10-01 · 本地历史证据与首版测量合同', ['研究问题', '本轮事实', '含义'], [
  ['是否已建立正净期望', '尚未建立', '同步估值、可见盘口与成交证据未齐。'],
  ['原始档案', data.status.raw_observations, `覆盖 ${data.status.distinct_raw_days} 个日期；条数含不同来源保存记录。`],
  ['合格 M1 收盘估值', 0, 'IOPV 外层时间不能证明估值经济时间。'],
  ['可识别 M2 事件', 0, '没有连续价差和固定价格范围内深度。'],
  ['有证据的成交', 0, '订单表无记录；无法据此估算成交率。'],
  ['费用后期望与夏普', '未计算', '不把无样本写成零收益或零风险。'],
  ['主账户 / 敏感性', '20 万元 / 2 万元', '采用研究假设；实际 ETF 佣金仍未知。'],
  ['M1 固定判断', '15:06', '只接纳此前收到且同步到 15:00 的估值。'],
  ['M2 固定观察', '冲击确认后 5 分钟', '未恢复及缺失终点均保留，观察中反弹不计利润。'],
  ['统一退出请求', '下一交易日 09:35', '请求后价格和成交才决定收益。'],
  ['执行优化 M3', '用途分类', '没有既定订单和基准成交就不能估算节省。'],
], [30, 29, 69]);
summary.getRange('A5:C15').format.rowHeight = 40;
summary.getRange('C5:C15').format.wrapText = true;
summary.getRange('B5:B15').format.horizontalAlignment='left';

const chinaClock=value=>value ? new Date(new Date(value).getTime()+8*60*60*1000) : null;
const syncRows = data.observations.map(r => [new Date(`${r.trade_date}T00:00:00Z`), r.source_timestamp_raw, chinaClock(r.received_at), r.price_observed ?? null,
  r.iopv_observed_unqualified ?? null, r.post_close_volume_shares ?? null, r.quality_status,
  r.quality_reasons, r.observation_id, r.raw_source_path, r.raw_source_sha256]);
const sync = table('同步行情', '同步行情表', '原值保留；经济时点、估值误差与排队字段缺失，全部未获准用于信号。',
  ['交易日期', '源时间原值', '接收时间（北京时间）', '所见价格（元）', 'IOPV 原值（元）', '盘后量原值（份）', '研究状态', '不合格原因', '记录编号', '本地来源', '原件 SHA-256'],
  syncRows, [14, 29, 40, 16, 18, 18, 15, 75, 57, 80, 70]);
sync.getRange(`D5:E${syncRows.length+4}`).setNumberFormat('0.0000');
sync.getRange(`A5:A${syncRows.length+4}`).setNumberFormat('yyyy-mm-dd');
sync.getRange(`C5:C${syncRows.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
sync.getRange(`F5:F${syncRows.length+4}`).setNumberFormat('#,##0');
sync.getRange(`H5:H${syncRows.length+4}`).format.wrapText = true;
sync.getRange(`A5:K${syncRows.length+4}`).format.rowHeight = 55;

table('事件', '事件表', '合格事件 0 条。缺失行情不等于当天没有冲击；逐日缺口见“日期覆盖”。',
  ['事件编号', '机制', '交易日期', '冲击起点', '固定观察终点', '恢复状态', '反对证据', '来源记录', '入场资格'], [], [23, 15, 15, 24, 26, 24, 32, 30, 27]);
table('订单', '订单表', '真实或可信重放订单 0 条。未提交委托，没有按盘后总量生成虚拟成交。',
  ['订单编号', '事件编号', '证据类型', '方向', '提交时间', '限价（元）', '申报数量（份）', '确认时间', '成交数量（份）', '成交均价（元）', '撤单确认时间', '状态', '证据位置'], [],
  [22, 22, 29, 12, 28, 18, 19, 28, 19, 19, 28, 25, 35]);
const result = table('结果', '结果表', '每个候选及账户规模均保留“未计算”；空白统计值表示缺少合格样本。',
  ['候选', '账户资金（元）', '计算状态', '合格事件', '已确认成交', '成交率', '净期望', '胜率', '盈亏比', '净夏普', '最大回撤', '完整账户'],
  data.results.map(r=>[r.variant, r.account_cny, r.status, r.qualifying_event_count, r.confirmed_filled_trade_count,
    r.fill_rate, r.net_expectancy, r.win_rate, r.payoff_ratio, r.net_sharpe, r.max_drawdown, r.full_account_ledger]),
  [43, 20, 59, 15, 17, 15, 15, 15, 15, 15, 16, 24]);
result.getRange('B5:B10').setNumberFormat('#,##0');
result.getRange('F5:H10').setNumberFormat('0.00%');
result.getRange('K5:K10').setNumberFormat('0.00%');
result.getRange('C5:C10').format.wrapText=true;
result.getRange('A5:L10').format.rowHeight=45;

const coverage = table('日期覆盖', '每日覆盖与缺失', '这 8 天均无法判定完整事件集合；不计为无冲击日、空仓日或合格交易日。',
  ['日期', '原始条数', 'M1 合格条数', 'M2 合格条数', 'M1 缺口', 'M2 缺口', '合同退出请求'],
  data.coverage.map(r=>[new Date(`${r.trade_date}T00:00:00Z`),r.raw_rows,r.qualified_m1_rows,r.qualified_m2_rows,'缺同步收盘估值及盘后队列','缺价差、深度及过去同刻基线',chinaClock(r.expected_exit_request_at)]),
  [16, 16, 18, 18, 37, 39, 34]);
coverage.getRange('A5:A12').setNumberFormat('yyyy-mm-dd');
coverage.getRange('G5:G12').setNumberFormat('yyyy-mm-dd hh:mm:ss');

const costs = table('费用演示', '费用门槛演示', '假设每边万二、每笔最低 5 元；摩擦仅用于示例，实际成交价格不得重复扣价差。',
  ['单边金额（元）', '每边佣金率', '最低佣金（元）', '每边佣金（元）', '往返佣金（bp）', '示例往返摩擦（bp）', '示例合计（bp）'],
  data.costs.map(r=>[r.notional_cny,.0002,5,null,null,10,null]), [23,22,22,23,25,28,25]);
for (let r=5;r<=7;r++) {
  costs.getRange(`D${r}`).formulas=[[`=MAX(A${r}*B${r},C${r})`]];
  costs.getRange(`E${r}`).formulas=[[`=2*D${r}/A${r}*10000`]];
  costs.getRange(`G${r}`).formulas=[[`=E${r}+F${r}`]];
}
costs.getRange('A5:A7').setNumberFormat('#,##0');
costs.getRange('B5:B7').setNumberFormat('0.00%');
costs.getRange('C5:G7').setNumberFormat('0.0');
costs.getRange('A5:C7').format.font.color='#1D4ED8';
costs.getRange('F5:F7').format.font.color='#1D4ED8';

const sourceRows = [
 ['2026 交易规则',data.sources.sse_rules,'15:05—15:30 收盘定价、时间优先；撤单须确认。'],
 ['交易日历',data.sources.sse_calendar,'2026-10-01 至 10-07 休市；下一交易日为 10-08。'],
 ['Level-1 定义',data.sources.sse_level1,'普通五档与累计成交信息不能替代个人排队记录。'],
 ['MD102 接口',data.sources.sse_md102_spec,'物理页 139—140：盘后未成交买卖总量；Timestamp 是最新成交时间。'],
 ['历史接口旧链接',data.sources.sse_history_spec,'本次 HTTP 404，保留失败原文；未取得该 PDF。'],
 ['ETF T+1',data.sources.sse_t1,'股票 ETF 当日新买入份额不能当日卖出。'],
 ['历史数据产品','https://www.sseinfo.com/services/assortment/historical/','官方产品含快照、逐笔和 IOPV；产品说明不代表已取得数据。'],
 ['本地分钟档案','510300_intraday_tencent_snapshots.parquet','267 行，2026-08-12 晚间接收。'],
 ['本地 IOPV 档案','510300_iopv_snapshots.parquet；510300_iopv_snapshots_v1_2.parquet','390 + 298 行，缺估值经济时点、误差和盘口。'],
 ['旧来源核查','510300_close_concession_source_feasibility_v1','2 条午休快照；279 个正数量成分的同步收盘价缺失。'],
];
const sources=table('数据来源','来源与口径','所有来源按其实际能力使用。原件、请求回执及完整字段存于研究报告目录。',['来源','网址或档案名','证据范围'],sourceRows,[30,105,75]);
sources.getRange('B5:C14').format.wrapText=true;
sources.getRange('A5:C14').format.rowHeight=62;

const inspection=await wb.inspect({kind:'table',range:'费用演示!A4:G7',include:'values,formulas',tableMaxRows:4,tableMaxCols:7,maxChars:2200});
await fs.writeFile(path.join(here,'费用公式检查.json'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'最终公式检查',maxChars:2000});
await fs.writeFile(path.join(here,'公式错误扫描.json'),errors.ndjson);
console.log(inspection.ndjson);
console.log(errors.ndjson);
for (const name of sheets) {
  const range = name==='研究结论' ? 'A1:C15' : name==='费用演示' ? 'A1:G8' : name==='同步行情' ? 'A1:H8' : name==='数据来源' ? 'A1:C8' : 'A1:G10';
  const rendered=await wb.render({sheetName:name,range,scale:1.35,format:'png'});
  await fs.writeFile(path.join(here,`${name}_预览.png`),new Uint8Array(await rendered.arrayBuffer()));
}
const resultFile=await SpreadsheetFile.exportXlsx(wb);
await resultFile.save(path.join(here,'510300_价格让步与压力恢复_四表.xlsx'));
console.log('已生成四表工作簿，并保存八张工作表预览与公式检查。');
