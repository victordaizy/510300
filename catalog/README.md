# 文件与附件索引

本目录是 `snapshot-2026-09-17` 的完整累计索引，覆盖 232,428 个原文件。附件可来自多个 Release；以每条附件的 URL 为准。

- [快照摘要](snapshot.json)：累计数量、增量数量、版本和盘点范围。
- [全部附件映射](assets.json)：每个附件的实际下载地址、大小、类型和 SHA-256。
- [本轮逐文件变化](changes-20260917.csv)：新增或修改、旧新摘要及最终存储位置。
- [增量比较回执](INCREMENTAL_COMPARISON.json)：当前纳入文件与历史索引的全量 SHA-256 比较。
- [研究目录](studies.csv)及[目录规模](directory_summary.json)。
- [排除范围](exclusions.json)及[保留历史缺失路径](RETAINED_HISTORICAL_PATHS.json)。
- [本轮附件全量回读](SAVED_ASSETS_VERIFICATION.json)：只针对新生成附件的全部原文件；[旧发布验证](history/snapshot-2026-09-09/SAVED_ASSETS_VERIFICATION.json)保留原版本范围。
- [忽略规则验证](IGNORE_RULES_VERIFICATION.json)和[最新汇总引用验证](STATUS_REFERENCES_VERIFICATION.json)；后者覆盖 418 个直接本地文件或目录引用。
- [本机环境快照](runtime.json)和[来源 Git 规则原件](source-control/README.md)。

## 原始文件清单

每个 CSV 最多 10,000 行，文件自身 SHA-256 由 `snapshot.json` 固定。

- [files-001.csv](files-001.csv)：10,000 个原文件。
- [files-002.csv](files-002.csv)：10,000 个原文件。
- [files-003.csv](files-003.csv)：10,000 个原文件。
- [files-004.csv](files-004.csv)：10,000 个原文件。
- [files-005.csv](files-005.csv)：10,000 个原文件。
- [files-006.csv](files-006.csv)：10,000 个原文件。
- [files-007.csv](files-007.csv)：10,000 个原文件。
- [files-008.csv](files-008.csv)：10,000 个原文件。
- [files-009.csv](files-009.csv)：10,000 个原文件。
- [files-010.csv](files-010.csv)：10,000 个原文件。
- [files-011.csv](files-011.csv)：10,000 个原文件。
- [files-012.csv](files-012.csv)：10,000 个原文件。
- [files-013.csv](files-013.csv)：10,000 个原文件。
- [files-014.csv](files-014.csv)：10,000 个原文件。
- [files-015.csv](files-015.csv)：10,000 个原文件。
- [files-016.csv](files-016.csv)：10,000 个原文件。
- [files-017.csv](files-017.csv)：10,000 个原文件。
- [files-018.csv](files-018.csv)：10,000 个原文件。
- [files-019.csv](files-019.csv)：10,000 个原文件。
- [files-020.csv](files-020.csv)：10,000 个原文件。
- [files-021.csv](files-021.csv)：10,000 个原文件。
- [files-022.csv](files-022.csv)：10,000 个原文件。
- [files-023.csv](files-023.csv)：10,000 个原文件。
- [files-024.csv](files-024.csv)：2,428 个原文件。

## 字段

| 字段 | 含义 |
|---|---|
| `path` | 相对原项目根目录的原路径 |
| `bytes`、`sha256` | 完整原文件的字节数和 SHA-256 |
| `storage` | `git` 或 `release` |
| `assets` | 还原所需附件名；分片按索引顺序拼接 |
| `reason` | 纳入理由 |
| `captured_mtime_ns` | 稳定读取的原文件修改时间 |
| `changed_since_inventory` | 本次读取时相对相应盘点记录是否变化；历史复用行保留其原值 |

附件可能还包含已经被新版本替代的旧成员；还原工具只恢复当前文件索引指定的成员。旧标签保存旧索引，因此历史版本仍可独立还原。
