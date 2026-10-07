import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const buildDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(buildDir, "../../..");
const outputDir = path.join(projectDir, "deliverables", "510300_训练数据与期权资料_20260924");
const workbookPath = path.join(outputDir, "510300_训练数据一览_含期权清单.xlsx");
const payload = JSON.parse(await fs.readFile(path.join(buildDir, "workbook_data.json"), "utf8"));
const workbook = Workbook.create();

function columnName(index) {
  let value = index + 1;
  let label = "";
  while (value > 0) {
    value -= 1;
    label = String.fromCharCode(65 + (value % 26)) + label;
    value = Math.floor(value / 26);
  }
  return label;
}

const verifications = [];
for (let sheetIndex = 0; sheetIndex < payload.sheets.length; sheetIndex += 1) {
  const item = payload.sheets[sheetIndex];
  const sheet = workbook.worksheets.add(item.name);
  const endColumn = columnName(item.headers.length - 1);
  const titleColumn = columnName(Math.min(item.headers.length, 8) - 1);
  const lastRow = item.rows.length + 4;
  sheet.showGridLines = false;
  sheet.tabColor = "#17665B";
  const all = sheet.getRange(`A1:${endColumn}${lastRow}`);
  all.format.font = { name: "Microsoft YaHei", size: 11, color: "#24342F" };
  all.format.rowHeight = 26;
  all.format.verticalAlignment = "center";
  for (let col = 0; col < item.headers.length; col += 1) {
    const letter = columnName(col);
    sheet.getRange(`${letter}1:${letter}${lastRow}`).format.columnWidth = item.widths[col];
  }
  sheet.getRange(`A1:${titleColumn}1`).merge();
  sheet.getRange("A1").values = [[item.title]];
  sheet.getRange(`A1:${endColumn}1`).format.font = { name: "Microsoft YaHei", size: 15, bold: true, color: "#155E52" };
  sheet.getRange(`A1:${endColumn}1`).format.rowHeight = 32;
  sheet.getRange(`A2:${titleColumn}2`).merge();
  sheet.getRange("A2").values = [[item.note]];
  sheet.getRange(`A2:${endColumn}2`).format.wrapText = true;
  sheet.getRange(`A2:${endColumn}2`).format.rowHeight = 42;
  sheet.getRange(`A2:${endColumn}2`).format.font = { name: "Microsoft YaHei", size: 10, color: "#596963" };
  sheet.getRange(`A3:${endColumn}3`).format.rowHeight = 8;
  const values = item.rows.map((row) => row.map((value, index) => {
    if (value !== null && item.dates.includes(item.headers[index])) {
      return new Date(`${String(value).slice(0, 10)}T00:00:00Z`);
    }
    if (typeof value === "boolean") return value ? "是" : "否";
    if (typeof value === "string" && value.startsWith("=")) return `'${value}`;
    return value;
  }));
  sheet.getRange(`A4:${endColumn}${lastRow}`).values = [item.headers, ...values];
  sheet.tables.add(`A4:${endColumn}${lastRow}`, true, `ResearchData${sheetIndex + 1}`);
  const header = sheet.getRange(`A4:${endColumn}4`);
  header.format.fill = "#E4F0EC";
  header.format.font = { name: "Microsoft YaHei", size: 11, bold: true, color: "#155E52" };
  header.format.wrapText = true;
  header.format.rowHeight = 38;
  sheet.freezePanes.freezeRows(4);
  for (let col = 0; col < item.headers.length; col += 1) {
    const label = item.headers[col];
    const letter = columnName(col);
    const cells = sheet.getRange(`${letter}5:${letter}${lastRow}`);
    if (item.dates.includes(label)) {
      cells.setNumberFormat("yyyy-mm-dd");
    } else if (item.percentages.includes(label)) {
      cells.setNumberFormat("0.00%;[Red](0.00%);0.00%");
    } else if (label.includes("价格") || label.endsWith("盘价") || label.endsWith("高价") || label.endsWith("低价") || label.includes("成交价") || label.includes("分红")) {
      cells.setNumberFormat("0.0000");
    } else if (/夏普|盈亏比|累积值/.test(label)) {
      cells.setNumberFormat("0.000");
    } else if (/元）/.test(label)) {
      cells.setNumberFormat("#,##0.00;[Red](#,##0.00);0.00");
    } else if (label.includes("年份")) {
      cells.setNumberFormat("0");
    } else if (/数|交易日/.test(label) && !label.includes("自然年")) {
      cells.setNumberFormat("#,##0");
    }
  }
  if (item.name === "数据说明") {
    sheet.getRange(`A5:B${lastRow}`).format.wrapText = true;
    sheet.getRange(`A5:B${lastRow}`).format.rowHeight = 44;
  }
  if (item.name === "期权数据清单") {
    sheet.getRange(`F5:F${lastRow}`).format.wrapText = true;
    sheet.getRange(`A5:F${lastRow}`).format.rowHeight = 46;
  }
  const actual = sheet.getRange(`A5:${endColumn}${lastRow}`).values;
  if (actual.length !== values.length) throw new Error(`行数不一致：${item.name}`);
  let numericChecks = 0;
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[r].length; c += 1) {
      if (typeof values[r][c] === "number") {
        if (typeof actual[r][c] !== "number" || Math.abs(actual[r][c] - values[r][c]) > 1e-10) {
          throw new Error(`数值不一致：${item.name}第${r + 5}行第${c + 1}列`);
        }
        numericChecks += 1;
      }
    }
  }
  verifications.push({ "工作表": item.name, "数据行数": values.length, "逐格数值检查数": numericChecks });
  console.log(`已完成工作表：${item.name}，${values.length}行`);
}

await fs.mkdir(path.join(buildDir, "previews"), { recursive: true });
for (const item of payload.sheets) {
  const lastColumn = columnName(Math.min(item.headers.length, 8) - 1);
  const bottomRow = Math.min(item.rows.length + 4, item.name === "数据说明" ? 10 : 12);
  const region = `A1:${lastColumn}${bottomRow}`;
  const preview = await workbook.render({ sheetName: item.name, range: region, scale: 1.3, format: "png" });
  await fs.writeFile(path.join(buildDir, "previews", `${item.name}.png`), new Uint8Array(await preview.arrayBuffer()));
  console.log(`已渲染检查区域：${item.name} ${region}`);
}
const inspection = await workbook.inspect({ kind: "region", sheetId: "三组主要结果", range: "A4:H7", maxChars: 4500, tableMaxRows: 4, tableMaxCols: 8 });
await fs.writeFile(path.join(buildDir, "workbook_inspection.json"), JSON.stringify(inspection, null, 2));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
await fs.writeFile(path.join(buildDir, "workbook_verification.json"), JSON.stringify(verifications, null, 2));
console.log(JSON.stringify({ "工作簿": workbookPath, "工作表": verifications }, null, 2));
