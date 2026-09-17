import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../..');
const latest = JSON.parse(await fs.readFile(path.join(root, 'reports/research/510300_daily_manual_signal_v1/latest.json'), 'utf8'));
const input = JSON.parse(await fs.readFile(path.join(root, `reports/research/510300_daily_manual_signal_v1/${latest.cutoff}/工作簿输入.json`), 'utf8'));
const wb = Workbook.create();
const summary = wb.worksheets.add('日常操作');
const daily = wb.worksheets.add('每日数据');
const finalRow = 1001;
const knownEnd = input.rows.length + 1;
const font = 'Microsoft YaHei';
const colors = { ink: '#172B4D', muted: '#526175', header: '#233B5D', pale: '#F3F6FA', input: '#FFF4CE', blue: '#1456A0', line: '#D4DDE7' };

for (const sheet of [summary, daily]) {
  sheet.showGridLines = false;
  const used = sheet === summary ? 'A1:F37' : `A1:N${finalRow}`;
  sheet.getRange(used).format.font = { name: font, size: 11, color: colors.ink };
  sheet.getRange(used).format.rowHeight = 22;
  sheet.getRange(used).format.verticalAlignment = 'center';
}
summary.tabColor = colors.header;
daily.tabColor = '#8FA2BA';
summary.getRange('A1:F37').format.columnWidth = 17;
summary.getRange('A1:A37').format.columnWidth = 26;
summary.getRange('B1:B37').format.columnWidth = 23;
summary.getRange('D1:D37').format.columnWidth = 28;
summary.getRange('E1:E37').format.columnWidth = 28;
summary.getRange('A2').values = [['510300 · 每日 D60 人工计算']];
summary.getRange('A2').format.font = { name: font, size: 16, bold: true, color: colors.ink };
summary.getRange('A3:F3').format.borders = { bottom: { style: 'thin', color: colors.line } };
summary.getRange('A5:B5').values = [['最近填写日期', null]];
summary.getRange('A6:A10').values = [['日内减隔夜标准分'], ['连续两日 > 1'], ['连续两日 < 0'], ['当天输入状态'], ['下一条填写位置']];
summary.getRange('D5:E5').values = [['每天只填四项', '填写方式']];
summary.getRange('D6:E9').values = [['日期', '仅交易日，按时间向下续填'], ['开盘价、收盘价', '不复权价格，单位元/份'], ['每份现金分红', '除息日填金额，其余填数字0'], ['填入位置', '“每日数据”的 A、C、D、E 列']];
summary.getRange('D11').values = [['B 列自动引用上一行实际收盘价。']];
summary.getRange('D12').values = [['从第二笔开始，不用重复填写昨收。']];
summary.getRange('D14').values = [['分红放在除息日，不放在派息到账日。']];
summary.getRange('D15').values = [['不得混用前复权价格与现金分红修正。']];
summary.getRange('D17').values = [['K 列是因子，L、M 列是因子门槛。']];
summary.getRange('D18').values = [['最新完整策略还有退出模型、仓位与状态。']];
summary.getRange('D19').values = [['本表不把单个因子门槛冒充最终买卖指令。']];
summary.getRange('A13:B13').values = [['固定口径', '现版数值']];
summary.getRange('A14:B17').values = [['累计及标准差窗口', 60], ['入场阈值（严格大于）', 1], ['差值退出阈值（严格小于）', 0], ['连续确认天数', 2]];
summary.getRange('A19:B19').values = [['数学关系', '']];
summary.getRange('A20').values = [['D = 日内对数收益 − 隔夜对数收益']];
summary.getRange('A21').values = [['F = SUM(D,60) / STDEV.S(D,60) / SQRT(60)']];
summary.getRange('A22').values = [['同一个 F = AVERAGE(D,60) / STDEV.S(D,60) × SQRT(60)']];
summary.getRange('A24').values = [['使用步骤']];
summary.getRange('A25').values = [['1. 在“每日数据”末尾黄色区域续填 A、C、D、E 四列，不插空行。']];
summary.getRange('A26').values = [['2. 收盘后使用完整日线。Excel 计算选项应为“自动”，也可按 F9。']];
summary.getRange('A27').values = [['3. 回到本页看最新分数与两日条件；缺数据或标准差为0时不产生条件。']];
summary.getRange('A28').values = [['4. 预留至第1001行；用完后复制最后一行 B、F:N 公式，并扩展本页引用。']];
summary.getRange('A30').values = [['已填历史与来源']];
summary.getRange('A31').values = [[`预填 ${input.rows[0].date} 至 ${input.cutoff} 共 ${input.rows.length} 个交易日。`]];
summary.getRange('A32').values = [['最初59行仅用于建立滚动窗口；第60个完整样本起产生分数。']];
summary.getRange('A33').values = [['来源：本地正式研究已接纳的不复权日线与官方分红表；同包附全历史 CSV。']];
summary.getRange('A34').values = [['标准差定义：https://support.microsoft.com/en-us/excel/functions/stdev-s-function']];
summary.getRange('A35').values = [['√60 只是尺度换算，不是年化，也不代表胜率或未来收益保证。']];
summary.getRange('A36').values = [['本表按本项目财富分解修正现金分红，现版结果逐项核对一致。']];
summary.getRange('B37').formulas = [[`=COUNT('每日数据'!$A$2:$A$${finalRow})+1`]];
summary.getRange('A37').values = [['最近数据行号']];
summary.getRange('B5').formulas = [[`=IF(B37<2,"",INDEX('每日数据'!$A$1:$A$${finalRow},B37))`]];
summary.getRange('B5').setNumberFormat('yyyy-mm-dd');
summary.getRange('B6').formulas = [[`=IF(B37<2,"待填写",IF(ISNUMBER(INDEX('每日数据'!$K$1:$K$${finalRow},B37)),INDEX('每日数据'!$K$1:$K$${finalRow},B37),"暂不可算"))`]];
summary.getRange('B6').setNumberFormat('0.000000');
summary.getRange('B7').formulas = [[`=IF(B37<2,"待填写",IF(ISNUMBER(INDEX('每日数据'!$L$1:$L$${finalRow},B37)),IF(INDEX('每日数据'!$L$1:$L$${finalRow},B37)=1,"满足入场因子条件","未满足"),"数据不足"))`]];
summary.getRange('B8').formulas = [[`=IF(B37<2,"待填写",IF(ISNUMBER(INDEX('每日数据'!$M$1:$M$${finalRow},B37)),IF(INDEX('每日数据'!$M$1:$M$${finalRow},B37)=1,"满足差值退出条件","未满足"),"数据不足"))`]];
summary.getRange('B9').formulas = [[`=IF(B37<2,"待填写",INDEX('每日数据'!$N$1:$N$${finalRow},B37))`]];
summary.getRange('B10').formulas = [['="每日数据，第"&(B37+1)&"行"']];

for (const region of ['A5:B5', 'D5:E5', 'A13:B13']) {
  summary.getRange(region).format.fill = colors.header;
  summary.getRange(region).format.font = { name: font, size: 11, bold: true, color: '#FFFFFF' };
  summary.getRange(region).format.rowHeight = 26;
}
summary.getRange('B6').format.font = { name: font, size: 16, bold: true, color: colors.header };
summary.getRange('A6:B10').format.rowHeight = 26;
summary.getRange('B14:B17').format.fill = colors.pale;
summary.getRange('A19:A19').format.font.bold = true;
summary.getRange('A24').format.font.bold = true;
summary.getRange('A30').format.font.bold = true;
summary.getRange('D6:E9').format.rowHeight = 26;
summary.getRange('D6:E9').format.wrapText = true;
summary.getRange('A31:A37').format.font.color = colors.muted;

daily.getRange('A1:N1').values = [['交易日期', '昨收（元/份）', '开盘（元/份）', '收盘（元/份）', '现金分红/份', '隔夜对数收益', '日内对数收益', '日内减隔夜', '60日差值和', '60日样本标准差', 'D60标准分', '两日>1 条件', '两日<0 条件', '输入状态']];
daily.getRange(`A1:N${finalRow}`).format.columnWidth = 16;
daily.getRange(`A1:A${finalRow}`).format.columnWidth = 15;
daily.getRange(`E1:E${finalRow}`).format.columnWidth = 16;
daily.getRange(`J1:J${finalRow}`).format.columnWidth = 20;
daily.getRange(`N1:N${finalRow}`).format.columnWidth = 26;
daily.getRange('A1:N1').format.fill = colors.header;
daily.getRange('A1:N1').format.font = { name: font, size: 11, bold: true, color: '#FFFFFF' };
daily.getRange('A1:N1').format.horizontalAlignment = 'center';
daily.getRange('A1:N1').format.rowHeight = 30;
daily.freezePanes.freezeRows(1);
daily.freezePanes.freezeColumns(1);
daily.getRange(`A2:E${knownEnd}`).values = input.rows.map(r => [new Date(`${r.date}T00:00:00Z`), r.previous_close, r.open, r.close, r.cash_dividend_per_share]);
daily.getRange(`A2:A${finalRow}`).setNumberFormat('yyyy-mm-dd');
daily.getRange(`B2:E${finalRow}`).setNumberFormat('0.000');
daily.getRange(`F2:J${finalRow}`).setNumberFormat('0.0000%');
daily.getRange(`K2:K${finalRow}`).setNumberFormat('0.000000');
daily.getRange(`L2:M${finalRow}`).setNumberFormat('0');
daily.getRange(`B2:M${finalRow}`).format.horizontalAlignment = 'right';
daily.getRange(`A2:A${finalRow}`).format.font.color = colors.blue;
daily.getRange(`C2:E${finalRow}`).format.font.color = colors.blue;
daily.getRange(`A${knownEnd+1}:A${finalRow}`).format.fill = colors.input;
daily.getRange(`C${knownEnd+1}:E${finalRow}`).format.fill = colors.input;
daily.getRange('B3').formulas = [['=IF(A3="","",IF(ISNUMBER(D2),D2,""))']];
daily.getRange(`B3:B${finalRow}`).fillDown();
daily.getRange('N2').formulas = [['=IF(A2="","",IF(COUNT(B2:E2)<>4,"补齐价格及分红",IF(OR(B2<=0,C2<=0,D2<=0,E2<0),"价格或分红不合法",IF(NOT(ISNUMBER(A2)),"日期须为Excel日期","已填完整"))))']];
daily.getRange('N3').formulas = [['=IF(A3="","",IF(COUNT(B3:E3)<>4,"补齐价格及分红",IF(OR(B3<=0,C3<=0,D3<=0,E3<0),"价格或分红不合法",IF(OR(NOT(ISNUMBER(A3)),NOT(ISNUMBER(A2)),A3<=A2),"日期须连续升序","已填完整"))))']];
daily.getRange(`N3:N${finalRow}`).fillDown();
daily.getRange('F2:H2').formulas = [['=IF(N2="已填完整",LN((C2+E2)/B2),"")', '=IF(N2="已填完整",LN((D2+E2)/(C2+E2)),"")', '=IF(COUNT(F2:G2)=2,G2-F2,"")']];
daily.getRange(`F2:H${finalRow}`).fillDown();
daily.getRange('I2:K60').values = [['']];
daily.getRange('I61:K61').formulas = [[
  '=IF(A61="","",IF(COUNT(H2:H61)=\'日常操作\'!$B$14,SUM(H2:H61),""))',
  '=IF(A61="","",IF(COUNT(H2:H61)=\'日常操作\'!$B$14,_xlfn.STDEV.S(H2:H61),""))',
  '=IF(COUNT(I61:J61)<>2,"",IF(J61=0,"",I61/(J61*SQRT(\'日常操作\'!$B$14))))'
]];
daily.getRange(`I61:K${finalRow}`).fillDown();
daily.getRange('L2:M61').values = [['']];
daily.getRange('L62:M62').formulas = [[
  '=IF(COUNT(K61:K62)<>2,"",IF(AND(K62>\'日常操作\'!$B$15,K61>\'日常操作\'!$B$15),1,0))',
  '=IF(COUNT(K61:K62)<>2,"",IF(AND(K62<\'日常操作\'!$B$16,K61<\'日常操作\'!$B$16),1,0))'
]];
daily.getRange(`L62:M${finalRow}`).fillDown();
daily.getRange(`N2:N${finalRow}`).conditionalFormats.add('containsText', {text:'补齐', format:{fill:'#FDE8E7',font:{color:'#9A241E'}}});
daily.getRange(`N2:N${finalRow}`).conditionalFormats.add('containsText', {text:'不合法', format:{fill:'#FDE8E7',font:{color:'#9A241E'}}});
daily.getRange(`N2:N${finalRow}`).conditionalFormats.add('containsText', {text:'日期须', format:{fill:'#FDE8E7',font:{color:'#9A241E'}}});
daily.getRange(`L62:M${finalRow}`).conditionalFormats.add('cellIs', {operator:'equal',formula:1,format:{fill:'#D9E7F5',font:{bold:true,color:'#233B5D'}}});
wb.recalculate();

const closeEnough = (actual, expected, label) => {
  if (typeof actual !== 'number' || Math.abs(actual - expected) > 1e-10) throw new Error(`${label} 核对失败：${actual} / ${expected}`);
};
const calculated = daily.getRange(`F61:M${knownEnd}`).values;
calculated.forEach((r, i) => {
  const expected = input.rows[i+59];
  ['overnight_log','intraday_log','difference','sum60','sample_std60','score60'].forEach((k,j)=>closeEnough(r[j],expected[k],`${expected.date} ${k}`));
  if (i > 0) {
    closeEnough(r[6],expected.entry_condition_2days,`${expected.date} 入场条件`);
    closeEnough(r[7],expected.difference_exit_condition_2days,`${expected.date} 退出条件`);
  }
});

// 在尚未导出的工作簿内临时测试续填、缺失与除息，随后完整恢复。
const nextRow = knownEnd + 1;
const previous = input.rows.at(-1).close;
const trial = new Date('2026-09-14T00:00:00Z');
daily.getRange(`A${nextRow}`).values = [[trial]];
daily.getRange(`C${nextRow}:E${nextRow}`).values = [[previous, previous + 0.01, null]];
wb.recalculate();
if (daily.getRange(`K${nextRow}`).values[0][0] !== '') throw new Error('缺失分红不可误当0');
daily.getRange(`E${nextRow}`).values = [[0]];
wb.recalculate();
const nextDifference = Math.log((previous+0.01)/previous);
const rolling = [...input.rows.slice(-59).map(r=>r.difference),nextDifference];
const mean = rolling.reduce((a,b)=>a+b,0)/60;
const std = Math.sqrt(rolling.reduce((a,b)=>a+(b-mean)**2,0)/59);
closeEnough(daily.getRange(`K${nextRow}`).values[0][0],mean/std*Math.sqrt(60),'续填下一交易日');
const dividendCase = input.rows.findIndex((r,i)=>i>59 && r.cash_dividend_per_share>0);
if (dividendCase<0) throw new Error('预填区缺少除息核对样本');
const dr = dividendCase+2;
closeEnough(daily.getRange(`H${dr}`).values[0][0],input.rows[dividendCase].difference,'除息日财富分解');
daily.getRange(`A${nextRow}`).values = [[null]];
daily.getRange(`C${nextRow}:E${nextRow}`).values = [[null,null,null]];
wb.recalculate();

await fs.writeFile(path.join(out,'公式计算核对.json'),JSON.stringify({status:'PASS_ARTIFACT_RECALCULATION',knownRows:input.rows.length,scoreRows:calculated.length,maxAllowedDifference:1e-10,checks:['与现版Python因子及条件逐项一致','现金分红除息日核对','下一交易日输入更新','分红留空不当成零'],nativeExcelOpened:false,latest:summary.getRange('A5:B10').values},null,2));
const errorScan = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:3000});
await fs.writeFile(path.join(out,'公式错误扫描.json'),errorScan.ndjson);
console.log((await wb.inspect({kind:'table',range:'日常操作!A5:B10',include:'values,formulas',tableMaxRows:6,tableMaxCols:2,maxChars:2500})).ndjson);
for (const [sheetName,range,name] of [['日常操作','A1:F37','操作页预览.png'],['每日数据',`A${knownEnd-8}:N${knownEnd+3}`,'每日数据预览.png']]) {
  const rendered = await wb.render({sheetName,range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,name),new Uint8Array(await rendered.arrayBuffer()));
}
const result = await SpreadsheetFile.exportXlsx(wb);
await result.save(path.join(out,'510300_D60每日手算.xlsx'));
console.log('Excel 已生成，已核对公式、连续续填、空白分红和除息日。');
