import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import { FileBlob, SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..', '..');
const config = JSON.parse(await fs.readFile(path.join(root, 'config', '510300_pressure_four_table_delivery_v1.json'), 'utf8'));
const report = path.join(root, config.report_directory);
const outputFile = path.join(here, config.workbook_filename);
try {
  await fs.access(outputFile);
  throw new Error('181日工作簿已存在，拒绝覆盖。');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
const payloadPath = path.join(report, 'workbook_data.json.gz');
const payloadBytes = await fs.readFile(payloadPath);
const data = JSON.parse(gunzipSync(payloadBytes).toString('utf8'));
if (process.argv.includes('--streaming-export')) {
  const python = path.join(os.homedir(), '.cache', 'codex-runtimes', 'codex-primary-runtime', 'dependencies', 'python', 'python.exe');
  const writer = path.join(root, 'scripts', 'export_510300_pressure_four_table_streaming_v1.py');
  const result = spawnSync(python, ['-X', 'utf8', writer], { cwd: root, stdio: 'inherit', windowsHide: true });
  if (result.error || result.status !== 0) throw result.error || new Error(`流式写出失败，退出码${result.status}。`);
  const previewPath = path.join(here, 'saved_workbook_viewports.xlsx');
  const savedViews = await SpreadsheetFile.importXlsx(await FileBlob.load(previewPath));
  const expected = [181, 543, 79288590, 47784, 0];
  const actual = savedViews.worksheets.getItem('研究状态').getRange('B6:B10').values.map(row => row[0]);
  if (actual.some((value, index) => value !== expected[index])) throw new Error('保存文件的覆盖缓存核对失败。');
  const inspected = await savedViews.inspect({ kind: 'table', range: '研究状态!A5:C18', include: 'values', tableMaxRows: 14, tableMaxCols: 3, maxChars: 5000 });
  await fs.writeFile(path.join(here, '研究状态公式检查.ndjson'), inspected.ndjson, 'utf8');
  const scanned = await savedViews.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 50 }, maxChars: 4000, summary: '保存页面及覆盖缓存错误检查' });
  await fs.writeFile(path.join(here, '公式错误扫描.ndjson'), scanned.ndjson, 'utf8');
  const ranges = { '研究状态': 'A1:C22', '同步行情': 'A1:H11', '事件': 'A1:F9', '订单': 'A1:F9', '结果': 'A1:H11',
    '日期覆盖': 'A1:G11', '数据来源': 'A1:G11', '字段说明': 'A1:B13', '公开来源': 'A1:D11' };
  for (const [name, range] of Object.entries(ranges)) {
    const preview = await savedViews.render({ sheetName: name, range, scale: 1.3, format: 'png' });
    await fs.writeFile(path.join(here, 'previews', `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
    console.log(`已检查最终保存文件页面：${name}。`);
  }
  console.log(`完整四表已保存并提取九页检查；保留${data.state.grid_rows.toLocaleString('zh-CN')}个网格，收益未计算。`);
  process.exit(0);
}
const wb = Workbook.create();
const font = 'Microsoft YaHei';
const names = ['研究状态', ...Object.keys(data.tables)];
for (const name of names) wb.worksheets.add(name);
const previews = path.join(here, 'previews');
await fs.mkdir(previews, { recursive: true });
console.log('工作簿创建开始：全部47,784个网格，缺失估值与收益保留空白。');

function columnName(number) {
  let value = number;
  let text = '';
  while (value > 0) {
    value -= 1;
    text = String.fromCharCode(65 + value % 26) + text;
    value = Math.floor(value / 26);
  }
  return text;
}

const widths = {
  '同步行情': [14, 15, 24, 20, 14, 16, 62, 23, 21, 23, 19, 24, 23, 21, 25, 21, 25, 20, 27, 26, 28, 23, 23, 23, 26, 26, 25, 24, 80],
  '事件': [25, 20, 15, 24, 24, 30, 45, 45, 30, 26, 30],
  '订单': [24, 24, 20, 34, 12, 24, 20, 24, 24, 24, 24, 24, 24, 30, 60],
  '结果': [46, 24, 20, 20, 23, 23, 23, 18, 18, 18, 21, 21, 21, 23, 21, 26, 58],
  '日期覆盖': [15, 16, 23, 25, 25, 22, 28, 29, 29, 29, 24, 38, 26],
  '数据来源': [27, 20, 15, 20, 23, 38, 23, 80],
  '字段说明': [39, 94],
  '公开来源': [30, 94, 30, 62],
};

function makeTable(name, descriptor) {
  const sh = wb.worksheets.getItem(name);
  const headers = descriptor.headers;
  const rows = descriptor.rows.map(row => row.map((value, index) => {
    if (value !== null && descriptor.date_columns.includes(index)) return new Date(`${value}T00:00:00Z`);
    return value;
  }));
  const finalRow = Math.max(rows.length + 5, 7);
  sh.showGridLines = false;
  // 大表正文沿用统一默认字体，避免为百万级单元格重复创建相同样式。
  sh.getRangeByIndexes(0, 0, 5, headers.length).format.font = { name: font, size: 10, color: '#27364A' };
  sh.getRange('A1').values = [[name === '同步行情' ? '510300 源行报价与时点资格' : `510300 ${name}`]];
  sh.getRange('A1').format.font = { name: font, size: 15, bold: true, color: '#172940' };
  sh.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeight = 30;
  sh.getRange('A2').values = [[descriptor.note]];
  sh.getRangeByIndexes(1, 0, 1, headers.length).format.rowHeight = 25;
  sh.getRange('A2').format.font = { name: font, size: 10, italic: true, color: '#526276' };
  const heading = sh.getRangeByIndexes(4, 0, 1, headers.length);
  heading.values = [headers];
  heading.format = { fill: '#263B55', font: { name: font, size: 10, bold: true, color: '#FFFFFF' },
    wrapText: true, verticalAlignment: 'center', horizontalAlignment: 'center', rowHeight: 43 };
  if (rows.length) {
    const body = sh.getRangeByIndexes(5, 0, rows.length, headers.length);
    body.values = rows;
    body.format.rowHeight = ['同步行情', '数据来源'].includes(name) ? 48 : 37;
    const wrapping = { '同步行情': [6, 28], '数据来源': [5, 7], '字段说明': [0, 1], '公开来源': [0, 1, 2, 3], '结果': [0, 15, 16] };
    for (const index of wrapping[name] || []) {
      const textColumn = sh.getRangeByIndexes(5, index, rows.length, 1);
      textColumn.format.wrapText = true;
      textColumn.format.verticalAlignment = 'center';
    }
    const table = sh.tables.add(`A5:${columnName(headers.length)}${rows.length + 5}`, true, `DeliveryTable${names.indexOf(name)}`);
    table.style = 'TableStyleLight9';
    for (const index of descriptor.date_columns) sh.getRangeByIndexes(5, index, rows.length, 1).setNumberFormat('yyyy-mm-dd');
    for (const index of descriptor.time_columns) sh.getRangeByIndexes(5, index, rows.length, 1).setNumberFormat('hh:mm');
  }
  for (let col = 0; col < headers.length; col++) {
    sh.getRangeByIndexes(0, col, finalRow, 1).format.columnWidth = widths[name][col];
  }
  if (rows.length > 20) sh.freezePanes.freezeRows(5);
  if (name === '同步行情') sh.freezePanes.freezeColumns(3);
  console.log(`已写入${name}：${rows.length.toLocaleString('zh-CN')}条。`);
  return sh;
}

for (const [name, descriptor] of Object.entries(data.tables)) makeTable(name, descriptor);
const sync = wb.worksheets.getItem('同步行情');
const gridRows = data.tables['同步行情'].rows.length;
sync.getRange(`I6:L${gridRows + 5}`).setNumberFormat('0');
sync.getRange(`M6:M${gridRows + 5}`).setNumberFormat('0.0000');
sync.getRange(`N6:N${gridRows + 5}`).setNumberFormat('0.000');
sync.getRange(`O6:O${gridRows + 5}`).setNumberFormat('#,##0.00');
sync.getRange(`P6:P${gridRows + 5}`).setNumberFormat('#,##0');
sync.getRange(`Q6:Q${gridRows + 5}`).setNumberFormat('0.000');
sync.getRange(`T6:U${gridRows + 5}`).setNumberFormat('0');
const result = wb.worksheets.getItem('结果');
result.getRange('B6:G11').setNumberFormat('#,##0');
result.getRange('H6:J11').setNumberFormat('0.00%');
result.getRange('K6:K11').setNumberFormat('0.000');
result.getRange('L6:L11').setNumberFormat('0.00%');
result.getRange('N6:N11').setNumberFormat('0.000');
result.getRange('O6:O11').setNumberFormat('0.00%');
wb.worksheets.getItem('日期覆盖').getRange('C6:J186').setNumberFormat('#,##0');
wb.worksheets.getItem('数据来源').getRange('G6:G548').setNumberFormat('#,##0');

const front = wb.worksheets.getItem('研究状态');
front.showGridLines = false;
front.getRange('A1:C22').format.font = { name: font, size: 11, color: '#27364A' };
front.getRange('A1:C22').format.rowHeight = 32;
front.getRange('A1:A22').format.columnWidth = 35;
front.getRange('B1:B22').format.columnWidth = 20;
front.getRange('C1:C22').format.columnWidth = 81;
front.getRange('A1').values = [['510300 压力修复研究四表']];
front.getRange('A1').format.font = { name: font, size: 16, bold: true, color: '#172940' };
front.getRange('A1:C1').format.rowHeight = 33;
front.getRange('A2').values = [['2026-01-05至09-30；181日最新来源覆盖，原M1/M2尚未准入。']];
front.getRange('A2').format.font = { name: font, size: 10, italic: true, color: '#526276' };
front.getRange('A5:C5').values = [['项目', '当前记录', '含义']];
front.getRange('A5:C5').format = { fill: '#263B55', font: { name: font, size: 11, bold: true, color: '#FFFFFF' }, rowHeight: 29 };
front.getRange('A6:C18').values = [
  ['来源日期数', null, '全部181日逐日覆盖；不是181个合格事件日。'],
  ['三流来源文件数', null, '行情、逐笔委托、逐笔成交各181文件。'],
  ['三流原始记录数', null, '原件清单行数之和，完整源文件身份见数据来源。'],
  ['测量网格数', null, '全部264格/日，含连续分钟及收盘、盘后覆盖探针。'],
  ['已准入原时点的网格数', null, '0表示原字段/时点门槛未通过，不表示市场没有机会。'],
  ['M1合格参考日期数', 0, '正IOPV字段仅4日，单位/经济时点/历史可得性未验证。'],
  ['正式事件数', null, '检验未运行；空值未知，不按0次市场事件解释。'],
  ['已提交订单数', 0, '研究没有提交订单；订单表保留空表。'],
  ['已确认成交数', 0, '没有真实成交或独立验证重放证据。'],
  ['成交后净期望', null, '尚未计算；结果表保留3版本×2账户规模。'],
  ['20万元全账户夏普', null, '完整现金日、持仓与执行账本尚未运行。'],
  ['原研究目标', '未完成', '来源表与工作簿完成没有被记为策略有效。'],
  ['本次费用（美元）', 0, '使用已存资格矩阵，未发起新数据下载。'],
];
front.getRange('A6:C18').format.wrapText = true;
front.getRange('A6:C18').format.rowHeight = 38;
front.getRange('B6:B10').formulas = [
  ["=COUNTA('日期覆盖'!$A$6:$A$186)"],
  ["=COUNTA('数据来源'!$A$6:$A$548)"],
  ["=SUM('数据来源'!$G$6:$G$548)"],
  [`=COUNTA('同步行情'!$C$6:$C$${gridRows + 5})`],
  [`=COUNTIF('同步行情'!$F$6:$F$${gridRows + 5},\"QUALIFIED\")`],
];
front.getRange('B6:B14').setNumberFormat('#,##0');
front.getRange('B15').setNumberFormat('0.00%');
front.getRange('B16').setNumberFormat('0.000');
front.getRange('A20').values = [['缺失值、名义报价与正式事件的解释见“字段说明”。']];
front.getRange('A21').values = [['下一步：新增时点、估值或执行证据后，先验证小样本，再启动原固定实验。']];

const expected = [181, 543, 79288590, 47784, 0];
const actual = front.getRange('B6:B10').values.map(row => row[0]);
if (actual.some((value, index) => value !== expected[index])) {
  throw new Error(`源覆盖汇总公式不一致：${JSON.stringify(actual)}`);
}
const inspected = await wb.inspect({ kind: 'table', range: '研究状态!A5:C18', include: 'values,formulas',
  tableMaxRows: 14, tableMaxCols: 3, maxChars: 5000 });
await fs.writeFile(path.join(here, '研究状态公式检查.ndjson'), inspected.ndjson, 'utf8');
console.log(`覆盖公式核对：${JSON.stringify(actual)}。`);
const errorScan = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 50 }, maxChars: 4000, summary: '本次工作簿公式错误检查' });
await fs.writeFile(path.join(here, '公式错误扫描.ndjson'), errorScan.ndjson, 'utf8');
const renderRanges = { '研究状态': 'A1:C22', '同步行情': 'A1:H11', '事件': 'A1:F9', '订单': 'A1:F9',
  '结果': 'A1:H11', '日期覆盖': 'A1:G11', '数据来源': 'A1:G11', '字段说明': 'A1:B13', '公开来源': 'A1:D11' };
for (const name of names) {
  const preview = await wb.render({ sheetName: name, range: renderRanges[name], scale: 1.3, format: 'png' });
  await fs.writeFile(path.join(previews, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
  console.log(`已生成视觉检查图：${name}。`);
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputFile);
const xlsxBytes = await fs.readFile(outputFile);
await fs.writeFile(path.join(here, 'workbook_export_receipt.json'), JSON.stringify({
  saved_at: new Date().toISOString(), workbook: outputFile, bytes: xlsxBytes.length,
  sha256: createHash('sha256').update(xlsxBytes).digest('hex'),
  compressed_data_sha256: createHash('sha256').update(payloadBytes).digest('hex'),
  builder_sha256: createHash('sha256').update(await fs.readFile(fileURLToPath(import.meta.url))).digest('hex'),
  sheet_names: names, source_summary_formula_values: actual, formal_event_count: null,
  strategy_returns: 'NOT_COMPUTED', goal_achieved: false, new_network_requests: 0, fee_usd: 0,
}, null, 2) + '\n', 'utf8');
console.log(`181日四表工作簿已保存，${xlsxBytes.length.toLocaleString('zh-CN')}字节；原金融目标未完成。`);
