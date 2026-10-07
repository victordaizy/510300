import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const outputDir = path.dirname(fileURLToPath(import.meta.url));
const input = JSON.parse(await fs.readFile(path.join(outputDir, 'workbook_input.json'), 'utf8'));
const exportOnly = process.argv.includes('--export-only');
const definitionsPreviewOnly = process.argv.includes('--definitions-preview');
const workbook = Workbook.create();
const fontFamily = 'Arial';
const previewDir = path.join(outputDir, 'previews');
await fs.mkdir(previewDir, { recursive: true });

function columnName(index) {
  let value = index + 1;
  let result = '';
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + value % 26) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

const previewRanges = [];
let tableNumber = 0;
function sheetTable(name, note, headers, rows, firstWidth = 42) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [[name]];
  sheet.getRange('A2').values = [[note]];
  sheet.getRange('A3').values = [['空白表示缺失或不适用。同比用百分比数值，差值用百分点；PMI用指数点。']];
  const endColumn = columnName(headers.length - 1);
  const lastRow = rows.length + 5;
  sheet.getRange(`A5:${endColumn}${lastRow}`).values = [headers, ...rows];
  sheet.getRange(`A1:${endColumn}${lastRow}`).format.font = { name: fontFamily, size: 10, color: '#233443' };
  sheet.getRange(`A1:${endColumn}${lastRow}`).format.rowHeight = 24;
  sheet.getRange(`A1:${endColumn}${lastRow}`).format.columnWidth = 13;
  sheet.getRange(`A1:A${lastRow}`).format.columnWidth = firstWidth;
  sheet.getRange('A1').format.font = { name: fontFamily, size: 18, bold: true };
  sheet.getRange('A1').format.rowHeight = 32;
  sheet.getRange(`A2:${endColumn}3`).format.font = { name: fontFamily, size: 10, color: '#617283', italic: true };
  sheet.getRange(`A5:${endColumn}5`).format = {
    fill: '#324F65', font: { name: fontFamily, size: 10, bold: true, color: '#FFFFFF' },
    wrapText: true, rowHeight: 46, verticalAlignment: 'center', horizontalAlignment: 'center',
  };
  if (rows.length) {
    sheet.getRange(`A6:${endColumn}${lastRow}`).setNumberFormat('0.0');
    tableNumber += 1;
    sheet.tables.add(`A5:${endColumn}${lastRow}`, true, `NbsTable${tableNumber}`);
  }
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(1);
  previewRanges.push({ name, range: `A1:${columnName(Math.min(headers.length - 1, 12))}${Math.min(lastRow, 15)}` });
  return sheet;
}

// 所有引用的工作表先创建，缺失数据保持空白。
const yearHeaders = Array.from({ length: input.末年 - input.首年 + 1 }, (_, i) => input.首年 + i);
const displayPriority = ['工业增加值', '服务业生产', '社会消费品零售', '固定资产投资', '民间固定资产投资', '房地产开发投资', '新建商品房销售面积', '新建商品房销售额', '工业企业利润', '工业企业营业收入', '工业应收账款', '工业产成品库存', 'CPI', 'PPI', '全国城镇调查失业率', 'M2', 'M1', 'M0', '制造业PMI'];
const macroRecords = [...input.宏观同月].sort((a, b) => {
  const aRank = displayPriority.indexOf(a.指标);
  const bRank = displayPriority.indexOf(b.指标);
  return (aRank < 0 ? 100 : aRank) - (bRank < 0 ? 100 : bRank);
});
const macroRows = macroRecords.map(row => [
  row.指标, row.口径, row.最新月份,
  ...yearHeaders.map(year => row[`${year}年同月`]),
  row.上年同月数值, row.前月数值, null, null,
  row.最近三月平均, row.之前三月平均, null, row.此前同月中位数, row.此前同月样本数, row.官方来源,
]);
const same = sheetTable('多年同月', `${input.首年}—${input.末年}年各年同一个月份。累计指标采用同一截止月。`,
  ['指标', '统计口径', '比较月份', ...yearHeaders.map(y => `${y}年`), '上年同月', '前月', '同比读数较上年差', '较前月读数差', '最近三月均值', '之前三月均值', '三月均值差', '此前同月中位数', '同月样本数', '官方来源'], macroRows);
same.getRange(`B1:B${macroRows.length + 5}`).format.columnWidth = 32;
same.getRange(`C6:C${macroRows.length + 5}`).setNumberFormat('@');
same.getRange('A3').values = [['M1在2025年前为旧口径，此后为新口径。CPI为各基期公布读数，原始版本保留。']];
const latestYearColumn = columnName(3 + yearHeaders.length - 1);
const lastYearValueColumn = columnName(3 + yearHeaders.length);
const previousValueColumn = columnName(4 + yearHeaders.length);
const yoyDeltaColumn = columnName(5 + yearHeaders.length);
const monthDeltaColumn = columnName(6 + yearHeaders.length);
const recentMeanColumn = columnName(7 + yearHeaders.length);
const previousMeanColumn = columnName(8 + yearHeaders.length);
const meanDeltaColumn = columnName(9 + yearHeaders.length);

const yearPathHeaders = ['指标', '统计口径', '年份', '截至月', ...Array.from({ length: 12 }, (_, i) => `${i + 1}月`), '实际有值月数', '口径说明'];
const yearPathRows = input.年内路径.map(row => [row.指标, row.口径, row.年份, row.比较截至月, ...Array.from({ length: 12 }, (_, i) => row[`${i + 1}月`]), row.有值月份数, row.说明]);
const paths = sheetTable('各年月度路径', '逐年列示1—12月。月份同比的平均数不能当作累计增速。', yearPathHeaders, yearPathRows);
paths.getRange(`B1:B${yearPathRows.length + 5}`).format.columnWidth = 32;
paths.getRange(`C6:D${yearPathRows.length + 5}`).setNumberFormat('0');
paths.getRange(`Q6:Q${yearPathRows.length + 5}`).setNumberFormat('0');

const seasonalRows = input.季调环比.map(row => [row.指标, row.数据月份, row['季调环比%'], null, row.公布时间, row.官方来源]);
const seasonal = sheetTable('环比原始表', '每个月采用本次保存的最新公布修订值。C列为官方季调环比。',
  ['指标', '数据月份', '官方季调环比%', '三个月累计变化%', '公布时间', '官方来源'], seasonalRows);
seasonal.getRange(`B6:B${seasonalRows.length + 5}`).setNumberFormat('@');
seasonal.getRange(`C6:D${seasonalRows.length + 5}`).setNumberFormat('0.00');
seasonal.getRange(`E1:E${seasonalRows.length + 5}`).format.columnWidth = 25;
seasonal.getRange(`F1:F${seasonalRows.length + 5}`).format.columnWidth = 68;
const seasonalCell = new Map(input.季调环比.map((r, index) => [`${r.指标}|${r.数据月份}`, index + 6]));

const rawRows = input.宏观月度.map(row => [row.指标, row.口径, row.数据月份, row.数值, row.原值, row.来源类别, row.统计版本, row.官方来源]);
const raw = sheetTable('宏观原始读数', '主要指标2017年至今的公布读数。价格指数减100另列，原指数仍保留。',
  ['指标', '统计口径', '数据月份', '分析读数', '原始数值', '来源类别', '统计版本', '官方来源'], rawRows);
raw.getRange(`B1:B${rawRows.length + 5}`).format.columnWidth = 32;
raw.getRange(`C6:C${rawRows.length + 5}`).setNumberFormat('@');
raw.getRange(`G1:G${rawRows.length + 5}`).format.columnWidth = 54;
raw.getRange(`H1:H${rawRows.length + 5}`).format.columnWidth = 68;
const rawCell = new Map(input.宏观月度.map((r, index) => [`${r.指标}|${r.口径}|${r.数据月份}`, index + 6]));

const comparisonItems = [
  ['工业增加值', '当月实际同比', '累计实际同比', '季调'],
  ['社会消费品零售', '当月名义同比', '累计名义同比', '季调'],
  ['固定资产投资', null, '累计名义同比', '季调'],
  ['CPI', '当月同比', null, '未季调'],
  ['PPI', '当月同比', null, '未季调'],
];
const shortRows = comparisonItems.map(([label, monthlyBasis, cumulativeBasis, adjustment]) => {
  const summary = input.宏观同月.find(r => r.指标 === label && r.口径 === (monthlyBasis || cumulativeBasis));
  return [label, summary.最新月份, null, null, null, adjustment, null, null, null];
});
const short = sheetTable('同比与环比', '同比、季调环比、三个月累计变化分别列示。环比不由同比读数相除得到。',
  ['指标', '最新月份', '当月同比%', '累计同比%', '当月环比%', '环比口径', '三个月累计变化%', '之前三个月累计%', '三月累计变化差'], shortRows);
short.getRange(`B6:B${shortRows.length + 5}`).setNumberFormat('@');
short.getRange(`C6:I${shortRows.length + 5}`).setNumberFormat('0.00');
const annualMomHeaders = ['指标', '年份', '同月月份', '官方当月同比%', '官方累计同比%', '官方真实环比%', '环比口径', '三个月季调累计%', '官方来源'];
const annualMomRows = input.同比环比多年同月.map(r => [r.指标, r.年份, r.同月月份, r['官方当月同比%'], r['官方累计同比%'], r['官方真实环比%'], r.环比口径, r['三个月季调累计变化%'], r.官方来源]);
if (annualMomRows.length) {
  short.getRange('A12').values = [['各年份同一个月份的同比与真实环比']];
  short.getRange('A12').format.font = { name: fontFamily, size: 12, bold: true };
  short.getRange(`A14:I${annualMomRows.length + 14}`).values = [annualMomHeaders, ...annualMomRows];
  short.getRange(`A14:I${annualMomRows.length + 14}`).format.font = { name: fontFamily, size: 10, color: '#233443' };
  short.getRange('A14:I14').format = { fill: '#324F65', font: { name: fontFamily, size: 10, bold: true, color: '#FFFFFF' }, wrapText: true, rowHeight: 46 };
  short.getRange(`B15:B${annualMomRows.length + 14}`).setNumberFormat('0');
  short.getRange(`C15:C${annualMomRows.length + 14}`).setNumberFormat('@');
  short.getRange(`D15:H${annualMomRows.length + 14}`).setNumberFormat('0.00');
  short.tables.add(`A14:I${annualMomRows.length + 14}`, true, 'NbsAnnualRealMom');
  previewRanges.find(r => r.name === '同比与环比').range = 'A1:I20';
}

const industryRows = [...input.行业同期].sort((a, b) => b.数据月份.localeCompare(a.数据月份) || a.行业.localeCompare(b.行业, 'zh-CN')).map(r => [
  r.行业, r.数据月份, r['增加值累计同比%'], r['营业收入累计同比%'], r['利润累计同比%'],
  r.营业收入累计亿元, r.利润累计亿元, r.营业收入可比基数亿元, r.利润可比基数亿元,
  null, null, null, r['应收账款期末同比%'], r['产成品库存期末同比%'], r['利润数据库原始同比%'], r.利润比较说明,
  'https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData',
]);
const sectors = sheetTable('行业同期', '全部41行业各年1—8月。利润同比基数非正或缺失时单列，原始数据库百分比保留。',
  ['行业', '累计截止月份', '增加值累计同比%', '收入累计同比%', '可比利润累计同比%', '累计收入亿元', '累计利润亿元', '上年可比收入亿元', '上年可比利润亿元', '营业收入利润率%', '上年可比利润率%', '利润率同比差百分点', '应收账款同比%', '产成品库存同比%', '数据库原始利润同比%', '利润比较说明', '官方来源'], industryRows, 54);
sectors.getRange(`B6:B${industryRows.length + 5}`).setNumberFormat('@');
sectors.getRange(`F6:L${industryRows.length + 5}`).setNumberFormat('#,##0.00');
sectors.getRange(`P1:P${industryRows.length + 5}`).format.columnWidth = 54;
sectors.getRange(`Q1:Q${industryRows.length + 5}`).format.columnWidth = 68;

const breadthRows = input.行业分布.map(r => [r.数据月份, r.生产有值行业数, r.增加值累计增长行业数, r.利润可比行业数, r.利润增长行业数, r.产增利降行业数, r['利润增速中位数%'], r['增加值增速中位数%'], r['行业利润率中位数%']]);
const breadth = sheetTable('行业历年分布', '按行业数统计，各年同一累计区间。中位数与工业总量增速使用不同权重。',
  ['累计截止月份', '生产有值行业数', '生产增长行业数', '利润可比行业数', '利润增长行业数', '产增利降行业数', '利润增速中位数%', '生产增速中位数%', '行业利润率中位数%'], breadthRows, 23);
breadth.getRange(`A6:A${breadthRows.length + 5}`).setNumberFormat('@');
breadth.getRange(`B6:F${breadthRows.length + 5}`).setNumberFormat('0');

const amountRows = input.金额复合.map(r => [r.指标, r.口径, r.月份, r.单位, r.原始总量, r['1年前同月总量'], null, r['2年前同月总量'], null, r['3年前同月总量'], null, r['5年前同月总量'], null, r.说明, r.来源, r.上月原始总量, r.原始月度差额, r['原值比值环比%']]);
const amounts = sheetTable('货币与总量', '原始金额与数量同月比值，保留1、2、3、5年基数。不能替代官方可比同比。',
  ['指标', '口径', '数据月份', '单位', '本期原始总量', '一年前同月总量', '一年比值增速%', '两年前同月总量', '两年复合增速%', '三年前同月总量', '三年复合增速%', '五年前同月总量', '五年复合增速%', '比较说明', '来源', '上月原始总量', '原始月度差额', '未季调原值环比%'], amountRows);
amounts.getRange(`C6:C${amountRows.length + 5}`).setNumberFormat('@');
amounts.getRange(`E6:M${amountRows.length + 5}`).setNumberFormat('#,##0.00');
amounts.getRange(`N1:N${amountRows.length + 5}`).format.columnWidth = 75;
amounts.getRange(`P6:R${amountRows.length + 5}`).setNumberFormat('#,##0.00');
amounts.getRange('A3').values = [['M1跨2025年口径断点的多年比值留空；存量月度差额不等于新增信贷或资金流。']];

const definitionRows = input.来源.map(r => [r.category, r.name, r.unit, r.period_basis, r.catalogue_path, r.annotation || '', r.indicator_id,
  'https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData']);
const definitions = sheetTable('指标定义', 'CPI基期更新和M1口径变化分别标明。数据为当前下载版本。',
  ['类别', '指标名称', '单位', '统计期口径', '分类版本', '官方注释', '指标ID', '官方来源'], definitionRows, 24);
definitions.getRange(`B1:B${definitionRows.length + 5}`).format.columnWidth = 65;
definitions.getRange(`E1:F${definitionRows.length + 5}`).format.columnWidth = 90;
definitions.getRange(`G1:G${definitionRows.length + 5}`).format.columnWidth = 38;
definitions.getRange(`H1:H${definitionRows.length + 5}`).format.columnWidth = 68;
definitions.getRange(`E6:F${definitionRows.length + 5}`).format.wrapText = true;
definitions.getRange(`A6:H${definitionRows.length + 5}`).format.verticalAlignment = 'top';
for (let index = 0; index < definitionRows.length; index++) {
  const textLines = [definitionRows[index][4], definitionRows[index][5]].map(text => String(text).split('\n').reduce((count, part) => count + Math.max(1, Math.ceil(part.length / 44)), 0));
  definitions.getRange(`A${index + 6}:H${index + 6}`).format.rowHeight = Math.max(24, Math.max(...textLines) * 15 + 6);
}
previewRanges[previewRanges.length - 1].range = 'A1:F9';

// 多年同月表的差值从同一行输入计算，累计跨年项目已经在输入中保持不适用。
for (let index = 0; index < macroRows.length; index++) {
  const row = index + 6;
  same.getRange(`${yoyDeltaColumn}${row}`).formulas = [[`=IF(OR(${latestYearColumn}${row}="",${lastYearValueColumn}${row}=""),"",${latestYearColumn}${row}-${lastYearValueColumn}${row})`]];
  same.getRange(`${monthDeltaColumn}${row}`).formulas = [[`=IF(OR(${latestYearColumn}${row}="",${previousValueColumn}${row}=""),"",${latestYearColumn}${row}-${previousValueColumn}${row})`]];
  same.getRange(`${meanDeltaColumn}${row}`).formulas = [[`=IF(OR(${recentMeanColumn}${row}="",${previousMeanColumn}${row}=""),"",${recentMeanColumn}${row}-${previousMeanColumn}${row})`]];
}

function shiftedPeriod(value, months) {
  const year = Number(value.slice(0, 4));
  const month = Number(value.slice(4));
  const date = new Date(Date.UTC(year, month - 1 + months, 1));
  return `${date.getUTCFullYear()}${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
}
for (let index = 0; index < input.季调环比.length; index++) {
  const row = input.季调环比[index];
  const locations = [0, -1, -2].map(offset => seasonalCell.get(`${row.指标}|${shiftedPeriod(row.数据月份, offset)}`));
  if (locations.every(Boolean)) seasonal.getRange(`D${index + 6}`).formulas = [[`=((1+C${locations[0]}/100)*(1+C${locations[1]}/100)*(1+C${locations[2]}/100)-1)*100`]];
}
for (let index = 0; index < comparisonItems.length; index++) {
  const [label, monthlyBasis, cumulativeBasis, adjustment] = comparisonItems[index];
  const row = index + 6;
  const period = shortRows[index][1];
  for (const [col, basis] of [['C', monthlyBasis], ['D', cumulativeBasis]]) {
    const sourceRow = rawCell.get(`${label}|${basis}|${period}`);
    if (sourceRow) short.getRange(`${col}${row}`).formulas = [[`='宏观原始读数'!D${sourceRow}`]];
  }
  if (adjustment === '季调') {
    const sourceRow = seasonalCell.get(`${label}|${period}`);
    const previousQuarter = seasonalCell.get(`${label}|${shiftedPeriod(period, -3)}`);
    if (sourceRow) {
      short.getRange(`E${row}`).formulas = [[`='环比原始表'!C${sourceRow}`]];
      short.getRange(`G${row}`).formulas = [[`='环比原始表'!D${sourceRow}`]];
    }
    if (previousQuarter) short.getRange(`H${row}`).formulas = [[`='环比原始表'!D${previousQuarter}`]];
    if (sourceRow && previousQuarter) short.getRange(`I${row}`).formulas = [[`=G${row}-H${row}`]];
  } else {
    const sourceRow = rawCell.get(`${label}|当月环比，未季调|${period}`);
    if (sourceRow) short.getRange(`E${row}`).formulas = [[`='宏观原始读数'!D${sourceRow}`]];
  }
}
for (let index = 0; index < industryRows.length; index++) {
  const row = index + 6;
  sectors.getRange(`J${row}`).formulas = [[`=IF(OR(F${row}="",G${row}="",F${row}<=0),"",G${row}/F${row}*100)`]];
  sectors.getRange(`K${row}`).formulas = [[`=IF(OR(H${row}="",I${row}="",H${row}<=0),"",I${row}/H${row}*100)`]];
  sectors.getRange(`L${row}`).formulas = [[`=IF(OR(J${row}="",K${row}=""),"",J${row}-K${row})`]];
}
for (let index = 0; index < amountRows.length; index++) {
  const row = index + 6;
  const source = input.金额复合[index];
  for (const [years, base, result] of [[1, 'F', 'G'], [2, 'H', 'I'], [3, 'J', 'K'], [5, 'L', 'M']]) {
    if (source[`${years}年复合增速%`] != null) amounts.getRange(`${result}${row}`).formulas = [[`=IF(OR(E${row}="",${base}${row}="",E${row}<=0,${base}${row}<=0),"",((E${row}/${base}${row})^(1/${years})-1)*100)`]];
  }
}

// 原生曲线直接引用单元格，保存后的数据修改会更新图表。
const choices = [['工业增加值', '当月实际同比'], ['社会消费品零售', '当月名义同比'], ['固定资产投资', '累计名义同比'], ['房地产开发投资', '累计名义同比'], ['工业企业利润', '累计可比同比'], ['制造业PMI', '季调扩散指数']];
const allMonths = [...new Set(input.宏观月度.map(r => r.数据月份))].sort();
const chartSheet = sheetTable('历史曲线', `${input.首年}年至最新月份。各指标口径单列，原始缺失保留空白。`,
  ['月份', ...choices.map(([label]) => label)], allMonths.map(period => [`${period.slice(0, 4)}-${period.slice(4)}`, ...choices.map(() => null)]), 16);
for (let index = 0; index < allMonths.length; index++) {
  const period = allMonths[index];
  for (let series = 0; series < choices.length; series++) {
    const [label, basis] = choices[series];
    const sourceRow = rawCell.get(`${label}|${basis}|${period}`);
    if (sourceRow) chartSheet.getRange(`${columnName(series + 1)}${index + 6}`).formulas = [[`='宏观原始读数'!D${sourceRow}`]];
  }
}
for (let series = 0; series < choices.length; series++) {
  const col = columnName(series + 1);
  const chart = chartSheet.charts.add('line', [chartSheet.getRange(`A5:A${allMonths.length + 5}`), chartSheet.getRange(`${col}5:${col}${allMonths.length + 5}`)]);
  chart.title = `${choices[series][0]}：${choices[series][1]}`;
  chart.titleTextStyle.typeface = fontFamily;
  chart.titleTextStyle.fontSize = 13;
  chart.hasLegend = false;
  chart.xAxis = { axisType: 'textAxis', textStyle: { typeface: fontFamily, fontSize: 10 }, tickLabelInterval: 12 };
  chart.yAxis = { numberFormatCode: '0.0', numberFormatSourceLinked: false, textStyle: { typeface: fontFamily, fontSize: 10 } };
  const startRow = 5 + series * 20;
  chart.setPosition(`I${startRow}`, `U${startRow + 17}`);
}

const priceRows = input.价格滚动基数.map(r => [r.指标, r.月份, r['当月同比%'], r['前月同比%'], r['当月未季调环比%'], r['上年同月未季调环比%'], null, null]);
const price = sheetTable('价格与滚动基数', '同比变动同时受本月价格变化与上年同月退出影响；公式保留舍入残差。',
  ['指标', '月份', '本月同比%', '上月同比%', '本月环比%', '上年同月环比%', '关系式计算同比%', '公布值与计算残差'], priceRows, 20);
price.getRange(`B6:B${priceRows.length + 5}`).setNumberFormat('@');
for (let index = 0; index < priceRows.length; index++) {
  const row = index + 6;
  price.getRange(`G${row}`).formulas = [[`=((1+D${row}/100)*(1+E${row}/100)/(1+F${row}/100)-1)*100`]];
  price.getRange(`H${row}`).formulas = [[`=C${row}-G${row}`]];
}
price.getRange(`C6:H${priceRows.length + 5}`).setNumberFormat('0.000');

const quality = [];
for (const spec of previewRanges) {
  const inspected = await workbook.inspect({ kind: 'table', range: `${spec.name}!${spec.range}`, include: 'values,formulas', tableMaxRows: 10, tableMaxCols: 12, maxChars: 2500 });
  quality.push({ sheet: spec.name, inspection: inspected.ndjson });
  if (!exportOnly && (!definitionsPreviewOnly || spec.name === '指标定义') && (!process.argv.includes('--updated-previews') || ['多年同月', '指标定义', '价格与滚动基数', '同比与环比', '货币与总量'].includes(spec.name))) {
    const preview = await workbook.render({ sheetName: spec.name, range: spec.range, scale: 1.4, format: 'png' });
    await fs.writeFile(path.join(previewDir, `${spec.name}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
if (!exportOnly && !definitionsPreviewOnly && !process.argv.includes('--updated-previews')) {
  const chartPreview = await workbook.render({ sheetName: '历史曲线', range: 'I4:U23', scale: 1.4, format: 'png' });
  await fs.writeFile(path.join(previewDir, '历史曲线图表.png'), new Uint8Array(await chartPreview.arrayBuffer()));
}
const previewFiles = new Set(await fs.readdir(previewDir));
for (let series = 1; series < choices.length; series++) {
  if (exportOnly || definitionsPreviewOnly) continue;
  const fileName = `历史曲线_${series + 1}.png`;
  if (process.argv.includes('--updated-previews') && previewFiles.has(fileName)) continue;
  const start = 5 + series * 20;
  const preview = await workbook.render({ sheetName: '历史曲线', range: `I${start - 1}:U${start + 18}`, scale: 1.4, format: 'png' });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 100 }, summary: '公式错误检查', maxChars: 5000 });
quality.push({ formulaErrors: errors.ndjson });
await fs.writeFile(path.join(outputDir, 'workbook_verification.json'), JSON.stringify(quality, null, 2));
console.log(JSON.stringify({ 公式错误扫描: errors.ndjson }, null, 2));
const exported = await SpreadsheetFile.exportXlsx(workbook);
const destination = path.join(outputDir, '全国月度多年同比环比.xlsx');
await exported.save(destination);
console.log(JSON.stringify({ 工作簿: destination, 工作表数: previewRanges.length, 原生历史曲线数: choices.length }, null, 2));
